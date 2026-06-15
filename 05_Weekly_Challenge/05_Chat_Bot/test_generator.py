from generator import generate

print("=== Test 1 ===")
print(generate("오늘 날씨가", max_new_tokens=25))

print("\n=== Test 2 ===")
print(generate("안녕하세요 오늘 기분이", max_new_tokens=20))

print("\n=== Test 3 (stop 조건) ===")
print(generate("오늘 날씨가", max_new_tokens=30, stop_sequences=["좋아요"]))