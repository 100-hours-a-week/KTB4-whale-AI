"""
Test for Step A-2: Chunking (src/model/chunker.py)

검증 항목:
1. 정상 케이스: 의도한 chunk_size, chunk_overlap에 맞게 분할되는가
2. Overlap 검증: 인접한 두 chunk가 실제로 겹치는 부분을 공유하는가
3. 예외 케이스: chunk_overlap >= chunk_size일 때 ValueError가 발생하는가
4. 경계 케이스: 빈 텍스트, chunk_size보다 짧은 텍스트 처리
5. 실제 데이터 통합 검증: nimbusflow_manual.md가 정상적으로 분할되는가
"""

from pathlib import Path

import pytest

from model.chunker import chunk_fixed_size  # noqa: E402
from model.document_loader import load_document  # noqa: E402


class TestChunkFixedSize:
    """chunk_fixed_size() 함수에 대한 테스트 그룹"""

    def test_chunk_splits_into_expected_count(self):
        """정상 케이스: 텍스트 길이와 step(=chunk_size - chunk_overlap)으로 chunk 개수를 예측할 수 있어야 한다."""
        # 길이 25인 텍스트, chunk_size=10, chunk_overlap=0 -> step=10
        # 시작 위치: 0, 10, 20 -> 3개 chunk 예상 (각 10, 10, 5자)
        text = "a" * 25
        chunks = chunk_fixed_size(text, chunk_size=10, chunk_overlap=0)

        assert len(chunks) == 3
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
        assert len(chunks[2]) == 5

    def test_chunk_overlap_is_actually_shared_between_adjacent_chunks(self):
        """Overlap 검증: chunk_overlap만큼의 문자가 인접 chunk 사이에 실제로 겹쳐야 한다."""
        # 알아보기 쉬운 텍스트를 사용: 0~9 숫자를 반복
        text = "0123456789" * 3  # 길이 30
        chunk_size = 10
        chunk_overlap = 4

        chunks = chunk_fixed_size(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 첫 chunk의 마지막 chunk_overlap 글자가 두 번째 chunk의 시작 부분과 같아야 한다
        first_chunk_tail = chunks[0][-chunk_overlap:]
        second_chunk_head = chunks[1][:chunk_overlap]

        assert first_chunk_tail == second_chunk_head

    def test_overlap_greater_than_or_equal_to_chunk_size_raises_value_error(self):
        """예외 케이스: chunk_overlap >= chunk_size이면 무한 루프 위험이 있으므로 ValueError를 발생시켜야 한다."""
        text = "some sample text"

        with pytest.raises(ValueError):
            chunk_fixed_size(text, chunk_size=10, chunk_overlap=10)  # overlap == size

        with pytest.raises(ValueError):
            chunk_fixed_size(text, chunk_size=10, chunk_overlap=15)  # overlap > size

    def test_empty_text_returns_empty_list(self):
        """경계 케이스: 빈 텍스트를 넣으면 빈 리스트를 반환해야 한다 (에러가 나면 안 됨)."""
        chunks = chunk_fixed_size("", chunk_size=10, chunk_overlap=2)
        assert chunks == []

    def test_text_shorter_than_chunk_size_returns_single_chunk(self):
        """경계 케이스: 텍스트가 chunk_size보다 짧으면 chunk 1개만 반환해야 한다."""
        text = "short text"  # 10자
        chunks = chunk_fixed_size(text, chunk_size=300, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_real_nimbusflow_manual(self):
        """실제 데이터 통합 검증: nimbusflow_manual.md가 정상적으로 분할되고, 핵심 정보가 보존되는지 확인."""
        real_data_path = (
            Path(__file__).resolve().parent.parent / "data" / "nimbusflow_manual.md"
        )
        document = load_document(str(real_data_path))

        chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)

        # chunk가 1개 이상 생성되어야 한다
        assert len(chunks) > 0

        # 모든 chunk를 합쳤을 때(중복 제외하고 단순 검사) 핵심 키워드가
        # 적어도 하나의 chunk에는 보존되어 있어야 한다 (overlap 덕분에 경계에서 잘려도 살아남아야 함)
        combined_lower = " ".join(chunks).lower()
        assert "nimbusflow" in combined_lower
        assert "nf-227" in combined_lower
        assert "project driftwood" in combined_lower