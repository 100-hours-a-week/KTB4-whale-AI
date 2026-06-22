"""
Level 2: Answer Relevancy (가장 단순한 키워드 겹침 기반)

목표:
- 생성된 답변이 질문과 얼마나 관련 있는지를 가장 단순한 키워드 겹침 방식으로 측정

구현 방식:
- 질문과 답변을 단어 단위로 분리한 후 겹침 비율을 계산
- 겹침 비율이 threshold 이상이면 관련 있다고 판단
- 겹침 비율 자체를 Answer Relevancy 점수로 사용
"""

import re
from typing import Dict, Any


def preprocess_text(text: str) -> list[str]:
    """
    텍스트를 전처리하여 단어 리스트로 변환한다.
    - 소문자화
    - 특수문자 제거
    - 공백 기준 분리
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.split()


def evaluate_answer_relevancy(
    question: str,
    answer: str,
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Answer Relevancy를 가장 단순한 키워드 겹침 방식으로 평가한다.

    Args:
        question: 사용자 질문
        answer: 생성된 답변
        threshold: 관련 있다고 판단할 겹침 비율 기준 (기본값 0.3)

    Returns:
        {
            "answer_relevancy_score": float,   # 0.0 ~ 1.0
            "overlap_ratio": float,
            "is_relevant": bool,
            "threshold": float
        }
    """
    query_words = preprocess_text(question)
    answer_words = preprocess_text(answer)

    if not query_words:
        return {
            "answer_relevancy_score": 0.0,
            "overlap_ratio": 0.0,
            "is_relevant": False,
            "threshold": threshold
        }

    # 겹침 비율 계산
    query_set = set(query_words)
    overlap_count = sum(1 for word in answer_words if word in query_set)
    overlap_ratio = overlap_count / len(query_words)

    # 관련성 판단
    is_relevant = overlap_ratio >= threshold

    return {
        "answer_relevancy_score": round(overlap_ratio, 4),
        "overlap_ratio": round(overlap_ratio, 4),
        "is_relevant": is_relevant,
        "threshold": threshold
    }


if __name__ == "__main__":
    # Level 2 테스트용 예시
    question = "What is the default API port for NimbusFlow?"
    answer = "The default API port for NimbusFlow is 8842."

    result = evaluate_answer_relevancy(question, answer, threshold=0.3)

    print("=" * 60)
    print(f"[Answer Relevancy Score] {result['answer_relevancy_score']}")
    print(f"[Overlap Ratio] {result['overlap_ratio']}")
    print(f"[Is Relevant] {result['is_relevant']}")
    print(f"[Threshold] {result['threshold']}")
    print("=" * 60)