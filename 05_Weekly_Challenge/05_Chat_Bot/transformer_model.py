import torch.nn as nn
from positional_encoding import PositionalEncoding
from transformer_decoder import TransformerDecoder
from decoder_layer import DecoderLayer

class TransformerLanguageModel(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 1024,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model

        # 1. Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 3. Decoder Layer (Post-LN 스타일)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        # 4. TransformerDecoder (여러 층 쌓기)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        # 5. 출력층 (다음 토큰 예측)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, mask=None):
        """
        input_ids: (batch_size, seq_len)
        """
        # Embedding
        x = self.embedding(input_ids) * (self.d_model ** 0.5)  # 원본 논문에서 사용한 scaling 방법 (선택)

        # Positional Encoding
        x = self.pos_encoding(x)

        # Transformer Decoder
        x = self.decoder(x, mask)

        # 출력 (다음 토큰 확률 분포)
        logits = self.output_linear(x)

        return logits