# Scaled Dot Production Attention

## 입력(x)

## 생성 (Linear -> Q, Linear -> K, Linear -> L)

## Q * K^T -> 유사도 계산

## Scale (/sqrt(d_k)) -> 값 안정화

## Mask (Causal Mask) -> 미래 단어 차단

## Softmax -> 가중치(확률)로 변환

## Softmax 결과 * V -> 가중 평균

## Attention Output