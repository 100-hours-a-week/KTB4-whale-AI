"""
Test for Level 1: evaluate_context_precision() in debugs/evaluate_context_precision.py

검증 항목:
1. 정상 케이스: Context Precision 점수가 올바르게 계산되는가
2. 관련성 판단 검증: threshold에 따라 is_relevant이 올바르게 설정되는가
3. 빈 리스트 입력 시 처리
4. 점수 범위 검증 (0.0 ~ 1.0)
"""

import pytest

from debugs.evaluate_context_precision import evaluate_context_precision  # noqa: E402


class TestEvaluateContextPrecision:
    """evaluate_context_precision() 함수에 대한 테스트 그룹"""

    @pytest.fixture    
    def sample_data(self):
        """테스트용 샘플 데이터"""
        return {
            "question": "What is the default API port for NimbusFlow?",
            "retrieved_chunks": [
                "NimbusFlow exposes a REST API on port 8842 by default.",
                "The product was developed under the codename Project Driftwood.",
                "Users can configure the engine mode to solo, cluster, or hybrid_sync.",
                "The default checkpoint interval is 90 seconds."
            ]
        }

    def test_returns_score_in_valid_range(self, sample_data):
        """정상 케이스: 점수가 0.0 ~ 1.0 사이로 반환되는가"""
        result = evaluate_context_precision(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            threshold=0.3
        )
        assert 0.0 <= result["context_precision_score"] <= 1.0

    def test_judgments_have_is_relevant_field(self, sample_data):
        """각 chunk에 대한 판단 결과가 올바르게 포함되는가"""
        result = evaluate_context_precision(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            threshold=0.3
        )

        assert len(result["judgments"]) == len(sample_data["retrieved_chunks"])

        for judgment in result["judgments"]:
            assert "is_relevant" in judgment
            assert isinstance(judgment["is_relevant"], bool)
            assert "overlap_ratio" in judgment

    def test_empty_retrieved_chunks_returns_zero_score(self):
        """빈 리스트를 넣으면 점수가 0.0이 나와야 함"""
        result = evaluate_context_precision(
            question="dummy question",
            retrieved_chunks=[],
            threshold=0.3
        )
        assert result["context_precision_score"] == 0.0
        assert result["judgments"] == []

    def test_threshold_affects_relevant_count(self, sample_data):
        """threshold 값에 따라 관련 있다고 판단되는 개수가 달라지는가"""
        # threshold를 낮추면 더 많은 chunk가 관련 있다고 판단될 수 있음
        result_low = evaluate_context_precision(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            threshold=0.1
        )
        result_high = evaluate_context_precision(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            threshold=0.5
        )

        # threshold가 낮을수록 관련 있다고 판단되는 개수가 같거나 많아야 함
        relevant_low = sum(1 for j in result_low["judgments"] if j["is_relevant"])
        relevant_high = sum(1 for j in result_high["judgments"] if j["is_relevant"])

        assert relevant_low >= relevant_high