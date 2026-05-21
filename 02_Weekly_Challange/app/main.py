import json
from app.database import create_db_and_tables, engine
from app.models import Post
from app.crud.post import get_posts
from app.routers.posts import router as posts_router
from app.routers.comments import router as comments_router
from app.routers.summaries import router as summaries_router
from contextlib import asynccontextmanager
from fastapi import FastAPI
from ollama import chat
from sqlmodel import Session

# ====================== 시작 시 DB 초기화 & dummy.json 로드 ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    print("✅ DB 테이블 생성 완료")
    
    # dummy.json이 있으면 한 번만 로드
    try:
        with open('dummy.json', encoding="utf-8") as file:
            raw_data = json.load(file)
        
        with Session(engine) as session:
            # 이미 데이터가 있으면 스킵
            posts = get_posts(session)
            if not posts:
                for item in raw_data:
                    post = Post.model_validate(item)
                    session.add(post)
                session.commit()
                print("✅ dummy.json 데이터를 DB에 로드했습니다.")
    except FileNotFoundError:
        print("⚠️ dummy.json 파일이 없습니다. 빈 DB로 시작합니다.")

    yield # 이 시점부터 FastAPI 서버가 실제로 동작

app = FastAPI(title="커뮤니티 사이트 API", lifespan=lifespan)

app.include_router(posts_router)
app.include_router(comments_router)
app.include_router(summaries_router)

@app.get("/")
def root():
    return {"message": "커뮤니티 사이트 API 서버가 정상 작동 중입니다."}