"""
Step A-2: Chunking (Fixed-size 전략)
 
책임(Responsibility): raw text(string)를 받아 chunk 리스트(list[str])를 반환한다.
이 단계에서는 벡터화(embedding)를 하지 않는다.
 
전략: Fixed-size chunking
- 텍스트의 구조(헤더, 문단 등)를 고려하지 않고, 정해진 문자 수(chunk_size) 단위로 자른다.
- chunk_overlap을 두어, chunk 경계에서 문장이 잘려 의미가 손실되는 것을 완화한다.
"""

def chunk_fixed_size(
    raw_text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    raw text를 고정 길이(chunk_size) 단위로 분할한다. 인접 chunk는 chunk_overlap만큼 겹친다.

    Args:
        raw_text: Document Loading 단계에서 반환된 원본 텍스트
        chunk_size: 한 chunk에 포함될 최대 문자 수
        chunk_overlap: 인접한 두 chunk가 공유하는 문자 수

    Returns:
        고정 길이로 분할된 chunk 리스트

    Raises:
        ValueError: chunk_overlap이 chunk_size보다 크거나 같을 경우
                    (이 경우 step이 0 이하가 되어 무한 루프 발생)
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap({chunk_overlap})은 chunk_size({chunk_size})보다 작아야 합니다."
        )

    step = chunk_size - chunk_overlap  # 다음 chunk 시작 위치까지의 이동 거리

    chunks = []
    start = 0
    text_length = len(raw_text)

    while start < text_length:
        end = start + chunk_size
        chunk = raw_text[start:end].strip()

        if chunk:  # 공백만 남은 chunk는 제외
            chunks.append(chunk)

        start += step

    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from model.document_loader import load_document

    sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "nimbusflow_manual.md"
    document = load_document(str(sample_path))
    chunks = chunk_fixed_size(document, chunk_size=300, chunk_overlap=50)

    print(f"[Chunking 결과] 총 chunk 개수: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        preview = chunk[:60].replace("\n", " ")
        print(f"--- Chunk {i} (길이: {len(chunk)}자) ---\n{preview}...\n")