import random

def dummy_generate(prompt: str, max_length: int = 50) -> str:
    """
    단계 1용 Dummy Generator.
    나중에 이 함수를 진짜 모델 호출 코드로 완전히 교체할 예정입니다.
    """
    responses = [
        "알겠습니다. 더 자세히 말씀해 주세요.",
        "그렇군요. 계속 말씀해 주세요.",
        "흥미로운 이야기네요.",
        "네, 이해했습니다.",
    ]
    
    # prompt에 특정 단어가 있으면 그에 맞는 답변
    if "날씨" in prompt:
        base = "좋아서 외출하기 좋은 날씨예요."
    elif "안녕" in prompt or "반가" in prompt:
        base = "반가워요! 오늘 하루는 어떠세요?"
    else:
        base = random.choice(responses)
    
    # 간단한 길이 제한 (실제 모델에서는 더 정교하게 처리)
    full_text = prompt + " " + base
    if len(full_text) > max_length:
        full_text = full_text[:max_length]
    
    return full_text