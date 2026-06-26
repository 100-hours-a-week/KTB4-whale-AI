"""
Test for Step B: Prompt Augmentation (src/model/prompt_builder.py)

검증 항목:
1. 구조 검증: Instruction, Context, Question이 모두 prompt에 포함되는가
2. Context 번호 매김 검증: chunk 개수만큼 [Document N] 형태가 생성되는가
3. 내용 포함 검증: 각 chunk의 텍스트가 prompt 안에 그대로 들어가는가
4. 질문 포함 검증: 사용자 질문이 정확히 포함되는가
5. 예외 케이스: retrieved_chunks가 빈 리스트일 때 ValueError 발생 여부
"""

from pathlib import Path

import pytest

from rag_pipeline.prompt_builder import build_prompt  # noqa: E402


class TestBuildPrompt:
    """build_prompt() 함수에 대한 테스트 그룹"""

    def test_prompt_contains_instruction(self):
        """구조 검증: prompt에 '문서에 근거해 답하라'는 지시문이 포함되어야 한다."""
        question = "What is the default port?"
        retrieved_chunks = [("The default port is 8842.", 0.9)]

        prompt = build_prompt(question, retrieved_chunks)

        assert "based only on the provided documents" in prompt
        assert "cannot find the answer" in prompt

    def test_prompt_contains_question(self):
        """질문 포함 검증: 사용자 질문이 prompt에 정확히 포함되어야 한다."""
        question = "What is the internal codename of NimbusFlow?"
        retrieved_chunks = [("Some chunk text.", 0.8)]

        prompt = build_prompt(question, retrieved_chunks)

        assert question in prompt

    def test_each_chunk_is_labeled_with_document_number(self):
        """Context 번호 매김 검증: chunk 3개를 넣으면 [Document 1]~[Document 3]이 모두 생성되어야 한다."""
        question = "dummy question"
        retrieved_chunks = [
            ("first chunk text", 0.9),
            ("second chunk text", 0.8),
            ("third chunk text", 0.7),
        ]

        prompt = build_prompt(question, retrieved_chunks)

        assert "[Document 1]" in prompt
        assert "[Document 2]" in prompt
        assert "[Document 3]" in prompt

    def test_chunk_text_content_is_preserved(self):
        """내용 포함 검증: 각 chunk의 실제 텍스트가 prompt 안에 그대로 포함되어야 한다."""
        question = "dummy question"
        retrieved_chunks = [
            ("The internal codename is Project Driftwood.", 0.95),
            ("The default API port is 8842.", 0.85),
        ]

        prompt = build_prompt(question, retrieved_chunks)

        assert "The internal codename is Project Driftwood." in prompt
        assert "The default API port is 8842." in prompt

    def test_empty_retrieved_chunks_raises_value_error(self):
        """예외 케이스: retrieved_chunks가 빈 리스트이면 ValueError가 발생해야 한다."""
        question = "dummy question"

        with pytest.raises(ValueError):
            build_prompt(question, [])

    def test_prompt_with_real_nimbusflow_chunk(self):
        """실제 데이터 형태와 유사한 통합 검증: NimbusFlow 매뉴얼 스타일의 chunk로 prompt가 정상 조립되는지 확인."""
        question = "What is the internal codename of NimbusFlow during development?"
        retrieved_chunks = [
            (
                'The product\'s internal codename during development was "Project Driftwood."',
                0.81,
            ),
        ]

        prompt = build_prompt(question, retrieved_chunks)

        assert "Project Driftwood" in prompt
        assert question in prompt
        assert "[Document 1]" in prompt