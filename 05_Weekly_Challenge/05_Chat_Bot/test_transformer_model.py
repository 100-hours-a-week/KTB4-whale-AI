import torch
from transformer_model import TransformerLanguageModel

# 가상의 vocab_size
vocab_size = 1000

model = TransformerLanguageModel(
    vocab_size=vocab_size,
    d_model=64,
    num_heads=8,
    num_layers=4,
    d_ff=256
)

# 더미 입력 (batch=2, seq_len=10)
input_ids = torch.randint(0, vocab_size, (2, 10))

# 실행
logits = model(input_ids)
print("Logits shape:", logits.shape)   # 기대: (2, 10, 1000)