"""
Level 1: Context Recall (LLM-as-a-Judge 기반)

목표:
- 정답에 필요한 핵심 정보(Ground Truth) 각각이, 검색된 context로부터
  추론 가능한지를 LLM이 의미적으로 판단하여 Context Recall 점수를 계산

Level 0과의 차이:
- Level 0(키워드 기반)은 Ground Truth 문장의 단어가 context에 그대로
  등장하는지만 보았다. Ground Truth가 한국어 문장으로 작성된 경우,
  영어 context와는 단어 자체가 겹치지 않아 항상 매칭에 실패하는 구조적
  한계가 있었다.
- Level 1은 LLM에게 "이 Ground Truth 정보가 context로부터 추론 가능한가"를
  언어 표현과 무관하게 의미적으로 판단하게 한다.
"""

from model.generator import TextGenerator


def evaluate_context_recall(
    question: str,
    retrieved_chunks: list[str],
    ground_truth: list[str],
) -> dict:
    """
    Context Recall을 LLM-as-a-Judge 방식으로 평가한다.

    각 Ground Truth 항목에 대해 "이 정보가 검색된 context로부터 추론
    가능한가"를 LLM이 Yes/No로 판단하고, 추론 가능하다고 판단된 항목의
    비율을 점수로 계산한다.

    Args:
        question: 사용자 질문
        retrieved_chunks: 검색된 context 리스트
        ground_truth: 정답에 필요한 핵심 정보 (문장 리스트)

    Returns:
        {
            "context_recall_score": float,  # 0.0 ~ 1.0
            "judgments": list[dict],        # 각 ground truth 항목에 대한 판단 결과
        }
    """
    if not ground_truth:
        return {
            "context_recall_score": 0.0,
            "judgments": [],
        }

    combined_context = "\n\n".join(retrieved_chunks)
    generator = TextGenerator()

    judgments = []
    matched_count = 0

    for i, gt_sentence in enumerate(ground_truth):
        prompt = f"""You are an evaluator that judges whether a piece of information can be inferred from a given context.

Context:
{combined_context}

Information to check: {gt_sentence}

Can the above information be inferred from the context, regardless of the language it is written in?
Answer with exactly one word: Yes or No.
"""
        response = generator.generate(prompt, max_new_tokens=10)
        is_matched = response.strip().lower().startswith("yes")

        if is_matched:
            matched_count += 1

        judgments.append({
            "ground_truth_index": i,
            "ground_truth": gt_sentence,
            "is_matched": is_matched,
            "raw_response": response.strip(),
        })

    context_recall_score = matched_count / len(ground_truth)

    return {
        "context_recall_score": round(context_recall_score, 4),
        "judgments": judgments,
    }


if __name__ == "__main__":
    # Level 1 테스트용 예시 (Level 0과 동일한 데이터로 비교 가능하게 유지)
    question = "What is the default API port for NimbusFlow?"
    retrieved_chunks = [
        "NimbusFlow exposes a REST API on port 8842 by default.",
        "The product was developed under the codename Project Driftwood.",
    ]
    ground_truth = [
        "API 포트는 8842이다",
        "NimbusFlow는 데이터 파이프라인 엔진이다",
    ]

    result = evaluate_context_recall(question, retrieved_chunks, ground_truth)

    print("=" * 60)
    print(f"[Context Recall Score] {result['context_recall_score']}")
    print("=" * 60)

    for j in result["judgments"]:
        status = "포함됨" if j["is_matched"] else "미포함"
        print(f"[{status}] {j['ground_truth']} (LLM 응답: {j['raw_response']!r})")