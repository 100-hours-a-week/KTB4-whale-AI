import torch
from model.transformer_model import TransformerLanguageModel

def test_transformer_language_model():
    print("=== TransformerLanguageModel 테스트 시작 ===\n")

    # 하이퍼파라미터 설정
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 4
    d_ff = 512
    max_len = 256

    print(f"설정: vocab_size={vocab_size}, d_model={d_model}, "
          f"num_heads={num_heads}, num_layers={num_layers}, d_ff={d_ff}\n")

    # 모델 생성
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=0.1
    )

    # === 테스트 1: Forward Pass ===
    print("[테스트 1] Forward Pass")
    batch_size = 2
    seq_len = 12
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_ids)  # mask=None

    print(f"Input shape : {input_ids.shape}")   # (2, 12)
    print(f"Logits shape: {logits.shape}")      # 기대: (2, 12, 1000)
    print()

    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Logits shape이 예상과 다릅니다. 예상: {(batch_size, seq_len, vocab_size)}, 실제: {logits.shape}"

    print("✅ 테스트 1 통과: Forward Pass 정상 동작\n")

    # === 테스트 2: 텍스트 생성 (Generate) ===
    print("[테스트 2] 텍스트 생성 (Generate)")
    start_token = torch.tensor([[1]])  # 시작 토큰

    generated = model.generate(
        input_ids=start_token,
        max_new_tokens=15,
        temperature=0.8,
        top_k=30
    )

    print(f"시작 토큰 shape     : {start_token.shape}")
    print(f"생성된 시퀀스 shape : {generated.shape}")  # 기대: (1, 16)
    print(f"생성된 토큰 ID     : {generated.tolist()[0]}")
    print()

    assert generated.shape[1] == 1 + 15, "생성된 시퀀스 길이가 예상과 다릅니다."

    print("✅ 테스트 2 통과: Generate 정상 동작\n")

    # === 테스트 3: Causal Mask 적용 ===
    print("[테스트 3] Causal Mask 적용")
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

    logits_with_mask = model(input_ids, mask=causal_mask)
    print(f"Masked Logits shape: {logits_with_mask.shape}")
    print()

    assert logits_with_mask.shape == (batch_size, seq_len, vocab_size), \
        "Mask 적용 시 Logits shape이 예상과 다릅니다."

    print("✅ 테스트 3 통과: Causal Mask 적용 정상 동작\n")

    # === 테스트 4: EOS 조기 종료 (Early Stopping) ===
    print("[테스트 4] EOS 조기 종료 (Early Stopping)")

    torch.manual_seed(42)  # 재현 가능한 결과를 위해 시드 고정

    eos_token_id = 7
    start_token_eos = torch.tensor([[1]])

    generated_with_eos = model.generate(
        input_ids=start_token_eos,
        max_new_tokens=15,
        temperature=0.8,
        top_k=30,
        eos_token_id=eos_token_id
    )

    generated_ids = generated_with_eos.tolist()[0]
    print(f"생성된 토큰 ID (eos_token_id={eos_token_id}): {generated_ids}")
    print(f"생성된 시퀀스 길이: {len(generated_ids)}")
    print()

    if eos_token_id in generated_ids[1:]:
        # eos가 등장했다면, 반드시 시퀀스의 마지막 토큰이어야 함 (그 즉시 멈췄어야 하므로)
        assert generated_ids[-1] == eos_token_id, \
            "eos_token_id가 등장했는데 그 이후에도 생성이 계속되었습니다 (조기 종료 실패)."
        assert len(generated_ids) <= 1 + 15, \
            "조기 종료가 적용됐는데도 시퀀스 길이가 max_new_tokens 한도를 넘었습니다."
        print("eos가 생성되어 조기 종료됨을 확인")
    else:
        # eos가 한 번도 안 나왔다면, 기존과 동일하게 max_new_tokens만큼 채워야 함
        assert len(generated_ids) == 1 + 15, \
            "eos가 등장하지 않았는데도 시퀀스 길이가 max_new_tokens와 다릅니다."
        print("이번 샘플링에서는 eos가 생성되지 않음 (정상 — 시드/샘플링에 따라 발생 가능)")

    print("\n✅ 테스트 4 통과: EOS 조기 종료 정상 동작\n")

    # === 테스트 4-1: EOS 조기 종료가 실제로 발생하는 경우 (결정적 재현) ===
    print("[테스트 4-1] EOS 조기 종료 - 실제 발생 케이스 (결정적 시드)")

    # vocab_size를 작게 줄여 eos가 등장할 확률을 높이고, 고정 시드로 재현 가능하게 함
    torch.manual_seed(0)
    small_model = TransformerLanguageModel(
        vocab_size=10, d_model=16, num_heads=2, num_layers=1, d_ff=32, max_len=64
    )
    small_model.eval()

    eos_token_id_small = 3
    generated_small = small_model.generate(
        input_ids=torch.tensor([[1]]),
        max_new_tokens=15,
        temperature=0.8,
        top_k=5,
        eos_token_id=eos_token_id_small
    )
    generated_small_ids = generated_small.tolist()[0]
    print(f"생성된 토큰 ID: {generated_small_ids}")
    print(f"생성된 시퀀스 길이: {len(generated_small_ids)}")

    assert eos_token_id_small in generated_small_ids, \
        "이 시드/설정에서는 eos가 반드시 등장해야 합니다. 모델 구조가 변경되었는지 확인하세요."
    assert generated_small_ids[-1] == eos_token_id_small, \
        "eos가 등장했는데 마지막 토큰이 아닙니다 (조기 종료 실패)."
    assert len(generated_small_ids) < 1 + 15, \
        "eos가 등장했는데도 max_new_tokens만큼 끝까지 생성되었습니다 (조기 종료 실패)."

    print("\n✅ 테스트 4-1 통과: EOS 등장 시 즉시 조기 종료됨을 결정적으로 확인\n")

    print("=== 모든 테스트 완료 ===")

if __name__ == "__main__":
    test_transformer_language_model()