"""
Level 0: RAGAS 없이 Faithfulness를 가장 단순한 형태로 평가하는 스크립트

Root from Expansion 원칙에 따라, RAG 파이프라인과의 통합 없이
평가 로직만 독립적으로 구현한다.
"""

from model.generator import TextGenerator


def evaluate_faithfulness(
    question: str,
    answer: str,
    retrieved_chunks: list[str],
) -> dict:
    """
    생성된 답변의 Faithfulness(충실도)를 평가한다.
    
    Args:
        question: 사용자 질문
        answer: RAG가 생성한 답변
        retrieved_chunks: 검색된 context 리스트

    Returns:
        {
            "faithfulness_score": float,  # 0.0 ~ 1.0
            "judgments": list[dict]       # 각 주장에 대한 판단 결과
        }
    """
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks가 비어 있습니다.")

    # context를 하나의 문자열로 합침
    context = "\n\n".join(retrieved_chunks)

    # 평가용 프롬프트 (가장 단순한 형태)
    prompt = f"""You are an evaluator that checks whether the given answer is faithful to the provided context.

                    Question: {question}

                    Context:
                    {context}

                    Answer: {answer}

                    Please break down the answer into individual claims and judge each claim.
                    For each claim, answer in the following format:
                    Claim: <claim text>
                    Judgment: Yes or No
                    Reason: <brief reason>

                    Only use information from the Context above. Do not use external knowledge.
                """

    generator = TextGenerator()
    response = generator.generate(prompt, max_new_tokens=300)

    # 간단한 파싱 (실제로는 더 견고한 파싱이 필요하지만, Level 0에서는 단순화)
    judgments = []
    lines = response.strip().split("\n")
    current_claim = None

    for line in lines:
        line = line.strip()
        if line.startswith("Claim:"):
            current_claim = line.replace("Claim:", "").strip()
        elif line.startswith("Judgment:") and current_claim:
            judgment = line.replace("Judgment:", "").strip().lower()
            is_supported = judgment == "yes"
            judgments.append({
                "claim": current_claim,
                "is_supported": is_supported,
                "judgment": judgment
            })
            current_claim = None

    # Faithfulness 점수 계산
    if judgments:
        supported_count = sum(1 for j in judgments if j["is_supported"])
        faithfulness_score = supported_count / len(judgments)
    else:
        faithfulness_score = 0.0

    return {
        "faithfulness_score": round(faithfulness_score, 3),
        "judgments": judgments,
        "raw_response": response
    }


if __name__ == "__main__":
    # Level 0 테스트용 하드코딩 데이터 (debug_retrieval.py에서 검증된 질문)
    question = "What is the internal codename of NimbusFlow during development?"
    answer = "The internal codename of NimbusFlow during development was Project Driftwood."
    retrieved_chunks = [
        "The product's internal codename during development was \"Project Driftwood.\"",
        "NimbusFlow is a lightweight data pipeline orchestration engine designed for small to mid-sized data teams.",
    ]

    result = evaluate_faithfulness(question, answer, retrieved_chunks)

    print("=" * 60)
    print(f"[Faithfulness Score] {result['faithfulness_score']}")
    print("=" * 60)
    for i, j in enumerate(result["judgments"], 1):
        print(f"\n[Claim {i}] {j['claim']}")
        print(f"Judgment: {j['judgment'].upper()}")
    print("\n[Raw Response]")
    print(result["raw_response"])