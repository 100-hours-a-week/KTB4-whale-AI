"""
Level 1: Answer Relevancy (LLM-as-a-Judge 기반)

목표:
- 생성된 답변이 질문에 직접적으로 대응하는지를 LLM이 의미적으로 판단하여
  Answer Relevancy 점수를 계산

Level 0과의 차이:
- Level 0(키워드 겹침)은 답변에 질문의 단어가 얼마나 등장하는지만 보았다.
  이 방식은 "질문의 단어를 그대로 반복하지만 실제로는 질문에 답하지 않는
  답변"에 높은 점수를 줄 수 있고, 반대로 "단어는 다르지만 정확히 답하는
  답변"에 낮은 점수를 줄 수 있다는 한계가 있다.
- Level 1은 LLM에게 "이 답변이 질문에 직접적으로 대응하는가"를 판단하게
  하여, 표면적 단어 반복이 아닌 의미적 관련성을 평가한다.
"""

from model.generator import TextGenerator


def evaluate_answer_relevancy(
    question: str,
    answer: str,
) -> dict:
    """
    Answer Relevancy를 LLM-as-a-Judge 방식으로 평가한다.

    Args:
        question: 사용자 질문
        answer: 생성된 답변

    Returns:
        {
            "answer_relevancy_score": float,  # 0.0 또는 1.0 (LLM의 Yes/No 판단)
            "is_relevant": bool,
            "raw_response": str,
        }
    """
    if not answer.strip():
        return {
            "answer_relevancy_score": 0.0,
            "is_relevant": False,
            "raw_response": "",
        }

    generator = TextGenerator()

    prompt = f"""You are an evaluator that judges whether an answer directly addresses a question.

                Question: {question}

                Answer: {answer}

                Does the above answer directly address the question?
                Answer with exactly one word: Yes or No.
                """
    response = generator.generate(prompt, max_new_tokens=10)
    is_relevant = response.strip().lower().startswith("yes")

    return {
        "answer_relevancy_score": 1.0 if is_relevant else 0.0,
        "is_relevant": is_relevant,
        "raw_response": response.strip(),
    }


if __name__ == "__main__":
    # Level 1 테스트용 예시 (Level 0과 동일한 데이터로 비교 가능하게 유지)
    question = "What is the default API port for NimbusFlow?"
    answer = "The default API port for NimbusFlow is 8842."

    result = evaluate_answer_relevancy(question, answer)

    print("=" * 60)
    print(f"[Answer Relevancy Score] {result['answer_relevancy_score']}")
    print(f"[Is Relevant] {result['is_relevant']}")
    print(f"[LLM Raw Response] {result['raw_response']!r}")
    print("=" * 60)