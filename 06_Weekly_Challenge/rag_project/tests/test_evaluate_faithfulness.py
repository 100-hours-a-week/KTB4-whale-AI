"""
Test for Level 0: evaluate_faithfulness() in debugs/evaluate_faithfulness.py

검증 항목:
1. 정상 케이스: Faithfulness 점수가 올바르게 계산되는가
2. 예외 케이스: retrieved_chunks가 비어있을 때 ValueError가 발생하는가
3. 점수 범위 검증: Faithfulness 점수가 0.0 ~ 1.0 사이인가
4. 판단 결과 파싱 검증: is_supported 값이 올바르게 설정되는가
"""

import pytest

from debugs.evaluate_faithfulness import evaluate_faithfulness  # noqa: E402


class TestEvaluateFaithfulness:
    """evaluate_faithfulness() 함수에 대한 테스트 그룹"""

    @pytest.fixture(scope="class")
    @classmethod
    def sample_data(cls):
        """테스트용 샘플 데이터 (debug_retrieval.py에서 검증된 내용 기반)"""
        return {
            "question": "What is the internal codename of NimbusFlow during development?",
            "answer": "The internal codename of NimbusFlow during development was Project Driftwood.",
            "retrieved_chunks": [
                "The product's internal codename during development was \"Project Driftwood.\"",
                "NimbusFlow is a lightweight data pipeline orchestration engine designed for small to mid-sized data teams.",
            ],
        }

    def test_returns_faithfulness_score_in_valid_range(self, sample_data):
        """정상 케이스: Faithfulness 점수가 0.0 ~ 1.0 사이로 반환되는가"""
        result = evaluate_faithfulness(
            question=sample_data["question"],
            answer=sample_data["answer"],
            retrieved_chunks=sample_data["retrieved_chunks"],
        )

        assert 0.0 <= result["faithfulness_score"] <= 1.0

    def test_judgments_contain_is_supported_field(self, sample_data):
        """판단 결과 검증: judgments 리스트에 is_supported 필드가 존재하는가"""
        result = evaluate_faithfulness(
            question=sample_data["question"],
            answer=sample_data["answer"],
            retrieved_chunks=sample_data["retrieved_chunks"],
        )

        assert len(result["judgments"]) > 0
        for judgment in result["judgments"]:
            assert "is_supported" in judgment
            assert isinstance(judgment["is_supported"], bool)

    def test_empty_retrieved_chunks_raises_value_error(self):
        """예외 케이스: retrieved_chunks가 비어있으면 ValueError가 발생해야 함"""
        with pytest.raises(ValueError):
            evaluate_faithfulness(
                question="dummy question",
                answer="dummy answer",
                retrieved_chunks=[],
            )

    def test_faithfulness_score_is_one_when_answer_is_fully_supported(self, sample_data):
        """
        의미적 검증: context와 완전히 일치하는 답변을 주면 
        Faithfulness 점수가 1.0에 가까워야 한다.
        (이번 테스트 데이터는 context와 answer가 잘 일치하므로 1.0이 나올 가능성이 높음)
        """
        result = evaluate_faithfulness(
            question=sample_data["question"],
            answer=sample_data["answer"],
            retrieved_chunks=sample_data["retrieved_chunks"],
        )

        # context와 answer가 거의 동일하므로 높은 점수가 나와야 함
        assert result["faithfulness_score"] >= 0.8