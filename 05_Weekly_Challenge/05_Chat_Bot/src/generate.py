from typing import Optional
import random

def generate(
    prompt: str,
    max_new_tokens: int = 30,
    stop_sequences: Optional[list[str]] = None
) -> str:
    """
    단계 2: 반복 호출 방식의 간이 생성기 (Dummy)

    TODO: (2026.06.19)
        - 이 함수 내부를 TransformerLanguageModel.generate() 기반으로 완전히 교체해야 함.
        - 현재는 규칙 기반 더미 로직이므로, 실제 모델 연결 후 삭제 또는 대폭 수정 예정.
    """
    if stop_sequences is None:
        stop_sequences = []

    current_text = prompt
    last_chunk = ""
    generated_tokens = 0

    for _ in range(max_new_tokens):
        # === 여기서 "다음 조각"을 생성 ===
        # 단계 2에서는 간단한 규칙/로직으로 다음 조각을 만듭니다.
        next_chunk = _get_next_chunk(current_text, last_chunk)

        current_text += next_chunk
        last_chunk = next_chunk
        generated_tokens += 1

        # 종료 조건 체크
        if _should_stop(current_text, stop_sequences):
            break

        # 너무 길어지면 강제 종료
        if generated_tokens >= max_new_tokens:
            break

    return current_text


def _get_next_chunk(current_text: str, last_chunk: str = "") -> str:
    """
    단계 2용 임시 로직.
    나중에 이 함수를 '모델이 다음 토큰을 예측하는 로직'으로 교체합니다.
    """
    text = current_text.lower()

    # 1. 특정 키워드에 따른 응답
    if "날씨" in text and "좋" not in text:
        return " 좋아서"
    elif "좋아서" in text and "산책" not in text:
        return " 산책하기"
    elif "산책하기" in text and "좋" not in text:
        return " 좋아요."

    # 2. 기본 응답 (다양하게 여러 개 준비)
    default_responses = [
        " 알겠습니다.",
        " 더 말씀해 주세요.",
        " 이해했습니다.",
        " 계속 말씀해 주세요.",
        " 네, 그렇군요.",
    ]

    # 마지막에 생성한 것과 같은 응답은 제외
    available = [r for r in default_responses if r != last_chunk]

    # 사용 가능한 응답이 없으면 전체에서 랜덤 선택
    if not available:
        available = default_responses

    return random.choice(available)


def _should_stop(current_text: str, stop_sequences: list[str]) -> bool:
    """특정 단어가 나오면 생성을 멈춤"""
    for seq in stop_sequences:
        if seq in current_text:
            return True
    return False