"""
Test for Level 1: evaluate_context_recall() in debugs/evaluate_context_recall.py

검증 항목:
1. 빈 Ground Truth 입력 시 처리 (LLM 호출 없이 즉시 0.0 반환되는가)
2. context에 명확히 존재하는 정보는 "포함됨"으로 판단하는가
3. context에 명확히 없는 정보는 "미포함"으로 판단하는가
4. 언어가 달라도(한국어 Ground Truth, 영어 context) 의미 기반으로 판단 가능한가
   (Level 0의 구조적 한계였던 부분의 회귀 테스트)
5. 점수 범위 검증 (0.0 ~ 1.0)
"""

import pytest

from debugs.evaluate_context_recall import evaluate_context_recall  # noqa: E402
from model.generator import TextGenerator  # noqa: E402


@pytest.fixture(scope="module")
def generator():
    """모듈 전체에서 TextGenerator를 한 번만 로딩하여 재사용한다."""
    return TextGenerator()


class TestEvaluateContextRecall:
    """evaluate_context_recall() 함수에 대한 테스트 그룹 (Level 1: LLM-as-a-Judge)"""

    def test_empty_ground_truth_returns_zero_score_without_llm_call(self):
        """예외 케이스: Ground Truth가 빈 리스트이면 LLM 호출 없이 즉시 0.0을 반환해야 한다."""
        result = evaluate_context_recall(
            question="dummy question",
            retrieved_chunks=["some context"],
            ground_truth=[],
            generator=None
        )
        assert result["context_recall_score"] == 0.0
        assert result["judgments"] == []

    def test_score_is_in_valid_range(self, generator):
        """정상 케이스: 점수가 0.0 ~ 1.0 사이로 반환되는가"""
        question = "What is the default API port for NimbusFlow?"
        retrieved_chunks = ["NimbusFlow exposes a REST API on port 8842 by default."]
        ground_truth = ["The default port is 8842.", "This information does not exist."]

        result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

        assert 0.0 <= result["context_recall_score"] <= 1.0

    def test_information_present_in_context_is_matched(self, generator):
        """
        의미적 검증: context에 명확히 존재하는 정보는 "포함됨"으로 판단되어야 한다.
        """
        question = "What is the default API port for NimbusFlow?"
        retrieved_chunks = ["NimbusFlow exposes a REST API on port 8842 by default."]
        ground_truth = ["The default port is 8842."]

        result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

        assert result["judgments"][0]["is_matched"] is True

    def test_information_absent_from_context_is_not_matched(self, generator):
        """
        의미적 검증: context에 전혀 없는 정보는 "미포함"으로 판단되어야 한다.
        (Level 1 실제 실행 결과에서 확인된 패턴: Level 0의 키워드 매칭 오탐을
         Level 1이 정확히 교정한 사례를 재현한다)
        """
        question = "What is the default API port for NimbusFlow?"
        retrieved_chunks = ["NimbusFlow exposes a REST API on port 8842 by default."]
        ground_truth = ["NimbusFlow is a data pipeline engine."]

        result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

        assert result["judgments"][0]["is_matched"] is False

    def test_korean_ground_truth_against_english_context(self, generator):
        """
        회귀 테스트: Ground Truth가 한국어, context가 영어인 경우에도
        LLM이 언어와 무관하게 의미를 기반으로 판단할 수 있어야 한다.
        (Level 0은 단어 자체가 겹치지 않아 항상 매칭에 실패하는 구조적
         한계가 있었다 — Level 1에서 이 한계가 해소되었는지 확인한다)
        """
        question = "What is the default API port for NimbusFlow?"
        retrieved_chunks = ["NimbusFlow exposes a REST API on port 8842 by default."]
        ground_truth = ["API 포트는 8842이다"]

        result = evaluate_context_recall(question, retrieved_chunks, ground_truth, generator)

        assert result["judgments"][0]["is_matched"] is True