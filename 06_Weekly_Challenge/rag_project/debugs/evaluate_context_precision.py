"""
Level 1: Context Precision (LLM-as-a-Judge 기반)

목표:
- 검색된 context(chunk) 각각이 질문에 답하는 데 실제로 필요한 정보인지를
  LLM이 의미적으로 판단하여 Context Precision 점수를 계산

Level 0과의 차이:
- Level 0(키워드 겹침)은 "단어가 같은가"만 보았기 때문에, 의미는 같지만
  표현이 다른 경우(예: 질문의 "포트"와 context의 "8842")를 관련 없다고
  잘못 판단할 수 있었다.
- Level 1은 LLM에게 "이 chunk가 질문에 답하는 데 필요한 정보를 담고 있는가"를
  직접 판단하게 하여, 표면적 단어 일치가 아닌 의미적 관련성을 평가한다.

evaluate_faithfulness.py와 동일한 패턴(prompt 구성 -> TextGenerator.generate()
-> 응답 파싱)을 따른다.
"""

from model.generator import TextGenerator


def evaluate_context_precision(
    question: str,
    retrieved_chunks: list[str],
    generator: TextGenerator,
) -> dict:
    """
    Context Precision을 LLM-as-a-Judge 방식으로 평가한다.

    각 chunk에 대해 "이 chunk가 질문에 답하는 데 필요한 정보를 담고 있는가"를
    LLM이 Yes/No로 판단하고, 관련 있다고 판단된 chunk의 비율을 점수로 계산한다.

    Args:
        question: 사용자 질문
        retrieved_chunks: 검색된 context 리스트
        generator: 평가에 사용할 TextGenerator 인스턴스. 호출하는 쪽에서
                   미리 생성하여 여러 평가 함수가 모델 로딩을 공유할 수 있게 한다.

    Returns:
        {
            "context_precision_score": float,  # 0.0 ~ 1.0
            "judgments": list[dict],           # 각 chunk에 대한 판단 결과
        }
    """
    if not retrieved_chunks:
        return {
            "context_precision_score": 0.0,
            "judgments": [],
        }

    judgments = []
    relevant_count = 0

    for i, chunk in enumerate(retrieved_chunks):
        prompt = f"""You are an evaluator that judges whether a piece of context is necessary to answer a question.

Question: {question}

Context:
{chunk}

Does the above context contain information that is necessary to answer the question?
Answer with exactly one word: Yes or No.
"""
        response = generator.generate(prompt, max_new_tokens=10)
        is_relevant = response.strip().lower().startswith("yes")

        if is_relevant:
            relevant_count += 1

        judgments.append({
            "chunk_index": i,
            "is_relevant": is_relevant,
            "raw_response": response.strip(),
        })

    context_precision_score = relevant_count / len(retrieved_chunks)

    return {
        "context_precision_score": round(context_precision_score, 4),
        "judgments": judgments,
    }


if __name__ == "__main__":
    # Level 1 테스트용 예시 (Level 0과 동일한 데이터로 비교 가능하게 유지)
    question = "What is the default API port for NimbusFlow?"
    retrieved_chunks = [
        "NimbusFlow exposes a REST API on port 8842 by default.",
        "The product was developed under the codename Project Driftwood.",
        "Users can configure the engine mode to solo, cluster, or hybrid_sync.",
        "The default checkpoint interval is 90 seconds.",
    ]

    generator = TextGenerator()
    result = evaluate_context_precision(question, retrieved_chunks, generator)

    print("=" * 60)
    print(f"[Context Precision Score] {result['context_precision_score']}")
    print("=" * 60)

    for j in result["judgments"]:
        status = "관련 있음" if j["is_relevant"] else "관련 없음"
        print(f"Chunk {j['chunk_index']}: {status} (LLM 응답: {j['raw_response']!r})")