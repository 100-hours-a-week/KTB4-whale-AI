import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, status, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional
from ollama import chat
from sqlmodel import SQLModel, Field, Relationship, Session, create_engine, select, delete

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
            if session.exec(select(Post)).first() is None:
                for item in raw_data:
                    post = Post.model_validate(item)
                    session.add(post)
                session.commit()
                print("✅ dummy.json 데이터를 DB에 로드했습니다.")
    except FileNotFoundError:
        print("⚠️ dummy.json 파일이 없습니다. 빈 DB로 시작합니다.")

    yield # 이 시점부터 FastAPI 서버가 실제로 동작

app = FastAPI(title="커뮤니티 사이트 API", lifespan=lifespan)

# ====================== DB 모델 (SQLModel) ======================
class Comment(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    email: EmailStr
    content: str
    created_at: datetime = Field(default_factory=datetime.now)

    post_id: int = Field(foreign_key="post.id")
    post: "Post" = Relationship(back_populates="comments")

class Post(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    email: EmailStr
    title: str
    content: str
    likes: int = Field(default=0)
    views: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    comments: List[Comment] = Relationship(back_populates="post")

# ====================== Pydantic 응답/요청 모델 ======================
# User
class UserBase(BaseModel):
    email: EmailStr
    nickname: str

class UserCreate(UserBase):
    password: str

class UserRead(UserBase):
    id: int
    created_at: datetime

# Comment
class CommentBase(BaseModel):
    content: str

class CommentCreate(CommentBase):
    email: EmailStr

class CommentUpdate(BaseModel):
    content: Optional[str] = None

class CommentRead(CommentBase):
    id: int
    email: EmailStr
    created_at: datetime

# Post
class PostBase(BaseModel):
    title: str
    content: str

class PostCreate(PostBase):
    email: EmailStr

class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class PostRead(PostBase):
    id: int
    email: EmailStr
    likes: int = 0
    views: int = 0
    created_at: datetime
    updated_at: datetime
    comments: List[CommentRead] = []

# Summary
class SummaryRead(BaseModel):
    summary: str = "요약 생성 불가"

# ====================== DB 설정 ======================
sqlite_file_name = "community.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# ====================== In-memory 저장소 ======================
# with open('dummy.json', encoding="utf-8") as file:
#     raw_data = json.load(file)
# posts: List[PostRead] = [PostRead.model_validate(item) for item in raw_data]

# if posts:
#     post_id_counter = max(post.id for post in posts) + 1
#     comment_id_counter = max(c.id for post in posts for c in post.comments) + 1
# else:
#     post_id_counter = 1
#     comment_id_counter = 1

# ====================== 게시글(Post) 엔드포인트 ======================
@app.get('/posts', response_model=List[PostRead])
def get_posts(session: Session = Depends(get_session)):
    """모든 게시글 목록"""
    posts = session.exec(select(Post)).all()
    return posts

@app.get('/posts/{post_id}', response_model=PostRead)
def get_post(post_id: int, session: Session = Depends(get_session)):
    """특정 게시글 조회"""
    post = session.get(Post, post_id)
    if post:
        return post
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@app.post('/posts', response_model=PostRead, status_code=status.HTTP_201_CREATED)
def create_post(new_post: PostCreate, session: Session = Depends(get_session)):
    """게시글 생성"""
    post = Post.model_validate(new_post)
    session.add(post)
    session.commit()
    session.refresh(post)
    return post

@app.patch('/posts/{post_id}', response_model=PostRead)
def update_post(post_id: int, update_post: PostUpdate, session: Session = Depends(get_session)):
    post = session.get(Post, post_id)

    if post:
        post.title = update_post.title if update_post.title is not None else post.title
        post.content = update_post.content if update_post.content is not None else post.content
        post.updated_at = datetime.now()
        session.add(post)
        session.commit()
        session.refresh(post)
        return post
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@app.delete('/posts/{post_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_post(post_id: int, session: Session = Depends(get_session)):
    post = session.get(Post, post_id)

    if post:
        session.delete(post)
        session.commit()
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# ====================== 댓글(Comment) 엔드포인트 ======================
@app.post('/posts/{post_id}/comments', response_model=CommentRead, status_code=status.HTTP_201_CREATED)
def create_comment(post_id: int, new_comment: CommentCreate, session: Session = Depends(get_session)):
    post = session.get(Post, post_id)

    if post:
        comment = Comment.model_validate(new_comment)
        comment.post_id = post_id
        session.add(comment)
        session.commit()
        session.refresh(comment)
        return comment
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
    

@app.patch("/posts/{post_id}/comments/{comment_id}", response_model=CommentRead)
def update_comment(post_id: int, comment_id: int, update_comment: CommentUpdate, session: Session = Depends(get_session)):
    comment = session.get(Comment, comment_id)
    if comment and comment.post_id == post_id:
        comment.content = update_comment.content if update_comment.content is not None else comment.content
        session.add(comment)
        session.commit()
        session.refresh(comment)
        return comment
    raise HTTPException(status_code=404, detail="댓글을 찾을 수 없습니다.")

@app.delete("/posts/{post_id}/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_comment(post_id: int, comment_id: int, session: Session = Depends(get_session)):
    comment = session.get(Comment, comment_id)
    if comment and comment.post_id == post_id:
        session.delete(comment)
        session.commit()
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# ====================== 조회수 & 좋아요 ======================
@app.patch("/posts/{post_id}/view")
def increment_view(post_id: int, session: Session = Depends(get_session)):
    post = session.get(Post, post_id)
    if post:
        post.views += 1
        session.add(post)
        session.commit()
        session.refresh(post)
        return {"post_id": post_id, "views": post.views}
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@app.patch("/posts/{post_id}/like")
def increment_like(post_id: int, session: Session = Depends(get_session)):
    post = session.get(Post, post_id)
    if post:
        post.likes += 1
        session.add(post)
        session.commit()
        session.refresh(post)
        return {"post_id": post_id, "likes": post.likes}
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# ====================== AI 요약 엔드포인트 ======================
@app.post('/posts/summary', response_model=SummaryRead)
def create_posts_summary(session: Session = Depends(get_session)):
    """1. 전체 게시글 목록 요약"""
    if not session.exec(select(Post)).first():
        return SummaryRead()

    posts = session.exec(select(Post)).all()
    post_text = '\n\n'.join(
        [f"제목: {post.title}\n내용: {post.content}" for post in posts]
    )

    prompt = f"""
				아래 게시글들을 종합해서 한국어로 자연스럽게 요약해주세요.
				전체적인 주제, 주요 의견, 분위기를 200자 이내로 간결하게 작성해주세요.

				{post_text}
			  """

    try:
        response = chat(model='gemma4:e4b', messages=[{'role': 'user', 'content': prompt}])
        summarized_posts = response.message.content
        print(summarized_posts)  # 디버깅용

        return SummaryRead(summary=summarized_posts) if summarized_posts is not None else SummaryRead()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama 호출 실패: {str(e)} (Ollama 서버가 실행 중인지 확인하세요)"
        )

@app.post('/posts/{post_id}/comments/summary', response_model=SummaryRead)
def create_comments_summary(post_id: int, session: Session = Depends(get_session)):
    """2. 특정 게시글의 댓글 목록 요약"""
    post = session.get(Post, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
    if not post.comments:
        return SummaryRead()   # 모델 기본값 사용

    comments_text = '\n\n'.join(
        [f"내용: {comment.content}" for comment in post.comments]
    )

    prompt = f"""
                아래 댓글들을 한국어로 자연스럽게 요약해주세요.
                주요 의견, 분위기를 200자 이내로 간결하게 작성해주세요.

                댓글 목록:
                {comments_text}
                """

    try:
        response = chat(model='gemma4:e4b', messages=[{'role': 'user', 'content': prompt}])
        summarized_comments = response.message.content
        print(summarized_comments)

        return SummaryRead(summary=summarized_comments) if summarized_comments is not None else SummaryRead()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama 호출 실패: {str(e)} (Ollama 서버가 실행 중인지 확인하세요)"
        )