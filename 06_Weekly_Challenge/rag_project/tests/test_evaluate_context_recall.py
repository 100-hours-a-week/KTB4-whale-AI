"""
Test for Level 0: evaluate_context_recall() in debugs/evaluate_context_recall.py

검증 항목:
1. 정상 케이스: Context Recall 점수가 올바르게 계산되는가
2. 모든 Ground Truth가 매칭되는 경우
3. 일부 Ground Truth만 매칭되는 경우
4. Ground Truth가 빈 리스트일 때 처리
5. Threshold에 따른 결과 변화
"""

import pytest

from debugs.evaluate_context_recall import evaluate_context_recall  # noqa: E402


class TestEvaluateContextRecall:
    """evaluate_context_recall() 함수에 대한 테스트 그룹"""

    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        return {
            "question": "What is the default API port for NimbusFlow?",
            "retrieved_chunks": [
                "NimbusFlow exposes a REST API on port 8842 by default.",
                "The product was developed under the codename Project Driftwood."
            ],
            "ground_truth": [
                "API 포트는 8842이다",
                "NimbusFlow는 데이터 파이프라인 엔진이다"
            ]
        }

    def test_returns_score_in_valid_range(self, sample_data):
        """정상 케이스: 점수가 0.0 ~ 1.0 사이로 반환되는가"""
        result = evaluate_context_recall(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            ground_truth=sample_data["ground_truth"],
            threshold=0.3
        )
        assert 0.0 <= result["context_recall_score"] <= 1.0

    def test_all_ground_truth_matched(self, sample_data):
        """모든 Ground Truth가 매칭되는 경우 점수가 1.0이 나와야 함"""
        result = evaluate_context_recall(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            ground_truth=sample_data["ground_truth"],
            threshold=0.3
        )
        assert result["context_recall_score"] == 1.0

    def test_partial_ground_truth_matched(self, sample_data):
        """일부 Ground Truth만 매칭되는 경우 점수가 중간값이 나와야 함"""
        partial_ground_truth = [
            "API 포트는 8842이다",
            "존재하지 않는 정보이다"  # 매칭되지 않음
        ]
        result = evaluate_context_recall(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            ground_truth=partial_ground_truth,
            threshold=0.3
        )
        assert result["context_recall_score"] == 0.5

    def test_empty_ground_truth_returns_zero_score(self, sample_data):
        """Ground Truth가 빈 리스트이면 점수가 0.0이 나와야 함"""
        result = evaluate_context_recall(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            ground_truth=[],
            threshold=0.3
        )
        assert result["context_recall_score"] == 0.0
        assert result["judgments"] == []

    def test_threshold_affects_result(self, sample_data):
        """Threshold 값에 따라 매칭 결과가 달라질 수 있는가"""
        result_low = evaluate_context_recall(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            ground_truth=sample_data["ground_truth"],
            threshold=0.1
        )
        result_high = evaluate_context_recall(
            question=sample_data["question"],
            retrieved_chunks=sample_data["retrieved_chunks"],
            ground_truth=sample_data["ground_truth"],
            threshold=0.9
        )

        # threshold가 낮을수록 매칭될 가능성이 높거나 같아야 함
        assert result_low["context_recall_score"] >= result_high["context_recall_score"]