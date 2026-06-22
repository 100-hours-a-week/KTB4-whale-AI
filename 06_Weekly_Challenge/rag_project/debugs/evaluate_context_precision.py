"""
Level 1: Context Precision (가장 단순한 키워드 겹침 기반)

목표:
- 검색된 context 중 질문과 키워드가 얼마나 겹치는지를 기준으로 관련성을 판단
- 관련 있는 context의 비율을 Context Precision 점수로 계산

구현 방식:
- 질문과 context를 단어 단위로 분리한 후 겹침 비율을 계산
- 일정 기준(threshold) 이상 겹치면 관련 있다고 판단
"""

import re
from typing import List, Dict, Any


def preprocess_text(text: str) -> List[str]:
    """
    텍스트를 전처리하여 단어 리스트로 변환한다.
    - 소문자화
    - 특수문자 제거
    - 공백 기준 분리
    """
    # 소문자 변환 + 알파벳과 공백만 남김
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return words


def calculate_overlap_ratio(query_words: List[str], chunk_words: List[str]) -> float:
    """
    질문 단어와 chunk 단어 간의 겹침 비율을 계산한다.
    """
    if not query_words:
        return 0.0

    query_set = set(query_words)
    overlap_count = sum(1 for word in chunk_words if word in query_set)
    return overlap_count / len(query_words)


def evaluate_context_precision(
    question: str,
    retrieved_chunks: List[str],
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Context Precision을 가장 단순한 키워드 겹침 방식으로 평가한다.

    Args:
        question: 사용자 질문
        retrieved_chunks: 검색된 context 리스트
        threshold: 관련 있다고 판단할 겹침 비율 기준 (기본값 0.3)

    Returns:
        {
            "context_precision_score": float,      # 0.0 ~ 1.0
            "judgments": List[Dict],               # 각 chunk에 대한 판단 결과
            "threshold": float
        }
    """
    if not retrieved_chunks:
        return {
            "context_precision_score": 0.0,
            "judgments": [],
            "threshold": threshold
        }

    # 질문 전처리
    query_words = preprocess_text(question)

    judgments = []
    relevant_count = 0

    for i, chunk in enumerate(retrieved_chunks):
        chunk_words = preprocess_text(chunk)
        overlap_ratio = calculate_overlap_ratio(query_words, chunk_words)

        is_relevant = overlap_ratio >= threshold

        if is_relevant:
            relevant_count += 1

        judgments.append({
            "chunk_index": i,
            "overlap_ratio": round(overlap_ratio, 4),
            "is_relevant": is_relevant
        })

    # Context Precision 점수 계산
    context_precision_score = relevant_count / len(retrieved_chunks)

    return {
        "context_precision_score": round(context_precision_score, 4),
        "judgments": judgments,
        "threshold": threshold
    }


if __name__ == "__main__":
    # Level 1 테스트용 예시
    question = "What is the default API port for NimbusFlow?"
    retrieved_chunks = [
        "NimbusFlow exposes a REST API on port 8842 by default.",
        "The product was developed under the codename Project Driftwood.",
        "Users can configure the engine mode to solo, cluster, or hybrid_sync.",
        "The default checkpoint interval is 90 seconds."
    ]

    result = evaluate_context_precision(question, retrieved_chunks, threshold=0.3)

    print("=" * 60)
    print(f"[Context Precision Score] {result['context_precision_score']}")
    print(f"[Threshold] {result['threshold']}")
    print("=" * 60)

    for j in result["judgments"]:
        status = "관련 있음" if j["is_relevant"] else "관련 없음"
        print(f"Chunk {j['chunk_index']}: overlap={j['overlap_ratio']:.2f} → {status}")