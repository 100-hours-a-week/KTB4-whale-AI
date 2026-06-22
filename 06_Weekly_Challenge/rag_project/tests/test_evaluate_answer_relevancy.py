"""
Test for Level 2: evaluate_answer_relevancy() in debugs/evaluate_answer_relevancy.py

검증 항목:
1. 정상 케이스: Answer Relevancy 점수가 올바르게 계산되는가
2. 관련성 판단 검증: is_relevant 값이 올바르게 설정되는가
3. 빈 답변 입력 시 처리
4. Threshold에 따른 결과 변화
5. 점수 범위 검증 (0.0 ~ 1.0)
"""

import pytest

from debugs.evaluate_answer_relevancy import evaluate_answer_relevancy  # noqa: E402


class TestEvaluateAnswerRelevancy:
    """evaluate_answer_relevancy() 함수에 대한 테스트 그룹"""

    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        return {
            "question": "What is the default API port for NimbusFlow?",
            "answer": "The default API port for NimbusFlow is 8842."
        }

    def test_returns_score_in_valid_range(self, sample_data):
        """정상 케이스: 점수가 0.0 ~ 1.0 사이로 반환되는가"""
        result = evaluate_answer_relevancy(
            question=sample_data["question"],
            answer=sample_data["answer"],
            threshold=0.3
        )
        assert 0.0 <= result["answer_relevancy_score"] <= 1.0

    def test_is_relevant_field_exists(self, sample_data):
        """is_relevant 필드가 올바르게 포함되는가"""
        result = evaluate_answer_relevancy(
            question=sample_data["question"],
            answer=sample_data["answer"],
            threshold=0.3
        )
        assert "is_relevant" in result
        assert isinstance(result["is_relevant"], bool)

    def test_empty_answer_returns_zero_score(self):
        """빈 답변을 넣으면 점수가 0.0이 나와야 함"""
        result = evaluate_answer_relevancy(
            question="What is the default API port?",
            answer="",
            threshold=0.3
        )
        assert result["answer_relevancy_score"] == 0.0
        assert result["is_relevant"] is False

    def test_threshold_affects_result(self, sample_data):
        """Threshold 값에 따라 is_relevant 결과가 달라질 수 있는가"""
        result_low = evaluate_answer_relevancy(
            question=sample_data["question"],
            answer=sample_data["answer"],
            threshold=0.1
        )
        result_high = evaluate_answer_relevancy(
            question=sample_data["question"],
            answer=sample_data["answer"],
            threshold=0.9
        )

        # threshold가 낮을수록 관련 있다고 판단될 가능성이 높거나 같아야 함
        assert result_low["is_relevant"] >= result_high["is_relevant"]