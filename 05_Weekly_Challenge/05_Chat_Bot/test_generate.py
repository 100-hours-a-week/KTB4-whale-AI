import torch
from transformer_model import TransformerLanguageModel

def test_generate():
    print("=== 텍스트 생성 테스트 시작 ===\n")

    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )

    input_ids = torch.tensor([[1]])

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        temperature=0.8,
        top_k=20
    )

    print(f"생성된 시퀀스 길이: {generated_ids.shape[1]}")
    print(f"생성된 토큰 ID: {generated_ids.tolist()[0]}")
    print("\n=== 생성 테스트 완료 ===")

if __name__ == "__main__":
    test_generate()