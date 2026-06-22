"""
Level 0: Context Recall (가장 단순한 키워드 기반)

목표:
- 정답에 필요한 핵심 정보(Ground Truth)가 검색된 context에 얼마나 포함되어 있는지 측정

구현 방식:
- Ground Truth를 문장 형태의 리스트로 정의
- 검색된 context를 하나로 합쳐서 키워드 기반으로 검색
- 각 Ground Truth 문장이 포함되어 있는지 판단
"""

import re
from typing import List, Dict, Any


def preprocess_text(text: str) -> List[str]:
    """텍스트를 소문자화하고 단어 리스트로 변환"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.split()


def evaluate_context_recall(
    question: str,
    retrieved_chunks: List[str],
    ground_truth: List[str],
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Context Recall을 가장 단순한 키워드 기반으로 평가한다.

    Args:
        question: 사용자 질문
        retrieved_chunks: 검색된 context 리스트
        ground_truth: 정답에 필요한 핵심 정보 (문장 리스트)
        threshold: 포함 여부를 판단할 키워드 겹침 비율 기준

    Returns:
        {
            "context_recall_score": float,
            "judgments": List[Dict],   # 각 ground truth 항목에 대한 판단 결과
            "threshold": float
        }
    """
    if not ground_truth:
        return {
            "context_recall_score": 0.0,
            "judgments": [],
            "threshold": threshold
        }

    # 검색된 context를 하나로 합침
    combined_context = " ".join(retrieved_chunks)
    context_words = set(preprocess_text(combined_context))

    judgments = []
    matched_count = 0

    for i, gt_sentence in enumerate(ground_truth):
        gt_words = preprocess_text(gt_sentence)

        if not gt_words:
            is_matched = False
        else:
            overlap_count = sum(1 for word in gt_words if word in context_words)
            overlap_ratio = overlap_count / len(gt_words)
            is_matched = overlap_ratio >= threshold

        if is_matched:
            matched_count += 1

        judgments.append({
            "ground_truth_index": i,
            "ground_truth": gt_sentence,
            "overlap_ratio": round(overlap_ratio, 4) if gt_words else 0.0,
            "is_matched": is_matched
        })

    context_recall_score = matched_count / len(ground_truth)

    return {
        "context_recall_score": round(context_recall_score, 4),
        "judgments": judgments,
        "threshold": threshold
    }


if __name__ == "__main__":
    # Level 0 테스트용 예시
    question = "What is the default API port for NimbusFlow?"
    retrieved_chunks = [
        "NimbusFlow exposes a REST API on port 8842 by default.",
        "The product was developed under the codename Project Driftwood."
    ]
    ground_truth = [
        "API 포트는 8842이다",
        "NimbusFlow는 데이터 파이프라인 엔진이다"
    ]

    result = evaluate_context_recall(question, retrieved_chunks, ground_truth, threshold=0.3)

    print("=" * 60)
    print(f"[Context Recall Score] {result['context_recall_score']}")
    print(f"[Threshold] {result['threshold']}")
    print("=" * 60)

    for j in result["judgments"]:
        status = "포함됨" if j["is_matched"] else "미포함"
        print(f"[{status}] {j['ground_truth']} (overlap={j['overlap_ratio']})")