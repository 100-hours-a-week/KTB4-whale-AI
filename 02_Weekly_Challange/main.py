import json
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional
from ollama import chat

app = FastAPI()

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

# ====================== In-memory 저장소 ======================
with open('dummy.json', encoding="utf-8") as file:
    raw_data = json.load(file)
posts: List[PostRead] = [PostRead.model_validate(item) for item in raw_data]
post_id_counter = 1
comment_id_counter = 1

# ====================== 게시글(Post) 엔드포인트 ======================
@app.get('/posts', response_model=List[PostRead])
def get_posts():
    """모든 게시글 목록"""
    return posts

@app.get('/posts/{post_id}', response_model=PostRead)
def get_post(post_id: int):
    """특정 게시글 조회"""
    for post in posts:
        if post.id == post_id:
            return post
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@app.post('/posts', response_model=PostRead, status_code=status.HTTP_201_CREATED)
def create_post(new_post: PostCreate):
    """게시글 생성"""
    global post_id_counter
    post = PostRead(
        id=post_id_counter,
        email=new_post.email,
        title=new_post.title,
        content=new_post.content,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        comments=[]
    )
    posts.append(post)
    post_id_counter += 1
    return post

@app.patch('/posts/{post_id}', response_model=PostRead)
def update_post(post_id: int, update_post: PostUpdate):
    """게시글 수정"""
    for post in posts:
        if post.id == post_id:
            if update_post.title is not None:
                post.title = update_post.title
            if update_post.content is not None:
                post.content = update_post.content
            post.updated_at = datetime.now()
            return post
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@app.delete('/posts/{post_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_post(post_id: int):
    """게시글 삭제"""
    global posts
    posts = [post for post in posts if post.id != post_id]
    return None

# ====================== 댓글(Comment) 엔드포인트 ======================
@app.post('/posts/{post_id}/comments', response_model=CommentRead, status_code=status.HTTP_201_CREATED)
def create_comment(post_id: int, new_comment: CommentCreate):
    """댓글 생성"""
    global comment_id_counter
    for post in posts:
        if post.id == post_id:
            comment = CommentRead(
                id=comment_id_counter,
                email=new_comment.email,
                content=new_comment.content,
                created_at=datetime.now()
            )
            post.comments.append(comment)
            comment_id_counter += 1
            return comment
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@app.patch("/posts/{post_id}/comments/{comment_id}", response_model=CommentRead)
def update_comment(post_id: int, comment_id: int, update_comment: CommentUpdate):
    """댓글 수정"""
    for post in posts:
        if post.id == post_id:
            for comment in post.comments:
                if comment.id == comment_id:
                    if update_comment.content is not None:
                        comment.content = update_comment.content
                    return comment
    raise HTTPException(status_code=404, detail="댓글을 찾을 수 없습니다.")

@app.delete("/posts/{post_id}/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_comment(post_id: int, comment_id: int):
    """댓글 삭제"""
    for post in posts:
        if post.id == post_id:
            post.comments = [c for c in post.comments if c.id != comment_id]
            return None
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# ====================== 조회수 & 좋아요 ======================
@app.patch("/posts/{post_id}/view")
def increment_view(post_id: int):
    for post in posts:
        if post.id == post_id:
            post.views += 1
            return {"post_id": post_id, "views": post.views}
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@app.patch("/posts/{post_id}/like")
def increment_like(post_id: int):
    for post in posts:
        if post.id == post_id:
            post.likes += 1
            return {"post_id": post_id, "likes": post.likes}
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# ====================== AI 요약 엔드포인트 ======================
@app.post('/posts/summary', response_model=SummaryRead)
def create_posts_summary():
    """1. 전체 게시글 목록 요약"""
    if not posts:
        return SummaryRead()

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
def create_comments_summary(post_id: int):
    """2. 특정 게시글의 댓글 목록 요약"""
    for post in posts:
        if post.id == post_id:
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
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")