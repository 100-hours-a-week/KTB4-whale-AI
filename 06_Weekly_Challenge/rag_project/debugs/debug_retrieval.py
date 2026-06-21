"""
디버깅 스크립트: "8842"를 포함한 chunk가 실제로 몇 번째 인덱스이고,
검색 질문에 대해 몇 위로 검색되는지 확인한다.

트러블슈팅 #8 검증: Fixed-size chunking과 Section-based chunking을
동일한 절차로 각각 실행하여, 두 전략의 "8842" chunk 순위를 비교한다.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from model.document_loader import load_document
from model.chunker import chunk_by_section, chunk_fixed_size
from model.embedder import TextEmbedder
from model.vector_store import InMemoryVectorStore
from model.retriever import cosine_similarity

document = load_document("data/nimbusflow_manual.md")
question = "What is the default API port for NimbusFlow?"
embedder = TextEmbedder()

print("########## [전략 1] Fixed-size chunking ##########\n")

chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)

# 1. "8842"를 포함한 chunk(들)의 인덱스를 모두 찾는다
target_indices = [i for i, c in enumerate(chunks) if "8842" in c]
print(f"[점검 1] '8842'를 포함한 chunk 인덱스: {target_indices}")
for i in target_indices:
    print(f"  - chunk[{i}]: {chunks[i]!r}")

print()

# 2. 전체 임베딩 및 저장
vectors = embedder.encode(chunks)
store = InMemoryVectorStore()
store.add(chunks, vectors)

# 3. 질문에 대해 모든 chunk의 유사도 점수와 순위를 계산
query_vector = embedder.encode([question])[0]

all_scores = []
for i in range(len(store)):
    score = cosine_similarity(query_vector, store.vectors[i])
    all_scores.append((i, score))

all_scores.sort(key=lambda pair: pair[1], reverse=True)

print(f"[점검 2] 질문: {question}")
print("[점검 2] 전체 chunk 순위 (인덱스, 점수):")
for rank, (index, score) in enumerate(all_scores, start=1):
    marker = " <== TARGET (8842 포함)" if index in target_indices else ""
    print(f"  Rank {rank}: chunk[{index}] score={score:.4f}{marker}")

print("\n\n########## [전략 2] Section-based chunking ##########\n")

chunks = chunk_by_section(document, chunk_size=300, chunk_overlap=50)

# 1. "8842"를 포함한 chunk(들)의 인덱스를 모두 찾는다
target_indices = [i for i, c in enumerate(chunks) if "8842" in c]
print(f"[점검 1] '8842'를 포함한 chunk 인덱스: {target_indices}")
for i in target_indices:
    print(f"  - chunk[{i}]: {chunks[i]!r}")

print()

# 2. 전체 임베딩 및 저장
vectors = embedder.encode(chunks)
store = InMemoryVectorStore()
store.add(chunks, vectors)

# 3. 질문에 대해 모든 chunk의 유사도 점수와 순위를 계산
query_vector = embedder.encode([question])[0]

all_scores = []
for i in range(len(store)):
    score = cosine_similarity(query_vector, store.vectors[i])
    all_scores.append((i, score))

all_scores.sort(key=lambda pair: pair[1], reverse=True)

print(f"[점검 2] 질문: {question}")
print("[점검 2] 전체 chunk 순위 (인덱스, 점수):")
for rank, (index, score) in enumerate(all_scores, start=1):
    marker = " <== TARGET (8842 포함)" if index in target_indices else ""
    print(f"  Rank {rank}: chunk[{index}] score={score:.4f}{marker}")