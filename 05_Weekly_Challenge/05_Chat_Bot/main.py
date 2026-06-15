from fastapi import FastAPI, HTTPException
from schemas import GenerateRequest, GenerateResponse
from generator import generate as generate_text

app = FastAPI(title="한국어 챗봇 API (단계 1 - Dummy)")

@app.get("/")
def root():
    return {"message": "단계 1: FastAPI + Dummy Generator 동작 중"}

@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="prompt는 비어있을 수 없습니다.")
    
    try:
        # generate 함수 호출 (max_new_tokens로 전달)
        result = generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_length,   # ← 여기 주의
            stop_sequences=None
        )
        return GenerateResponse(
            generated_text=result,
            prompt=request.prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"생성 중 오류가 발생했습니다: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "stage": "step1_dummy",
        "message": "Dummy Generator로 동작 중입니다."
    }