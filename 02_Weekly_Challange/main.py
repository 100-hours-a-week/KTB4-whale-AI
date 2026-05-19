from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional

app = FastAPI()

# User
class UserBase(BaseModel):
	email: EmailStr # 사용자 이메일
	nickname: str # 닉네임

class UserCreate(UserBase):
	password: str # 비밀번호

class UserRead(UserBase):
	id: int # 식별자
	created_at: datetime # 계정 생성일

# Comment
class CommentBase(BaseModel):
	content: str

class CommentCreate(CommentBase):
	email: EmailStr

class CommentUpdate(BaseModel):
	content: Optional[str] = None

class CommentRead(CommentBase):
	id: int # 식별자
	email: EmailStr # 사용자 이메일
	created_at: datetime


# Post
class PostBase(BaseModel):
	title: str # 제목
	content: str # 내용

class PostCreate(PostBase):
	email: EmailStr

class PostUpdate(BaseModel):
	title: Optional[str] = None
	content: Optional[str] = None

class PostRead(PostBase):
	id: int # 식별자
	email: EmailStr # 사용자 이메일
	likes: int # 좋아요 수
	views: int # 조회수
	created_at: datetime # 생성일
	updated_at: datetime # 수정일
	comments: List[CommentRead] # 댓글

# ====================== In-memory 저장소 ======================
posts: List[PostRead] = []
post_id_counter = 0
comment_id_counter = 0

# ====================== Post 엔드포인트 - 전체 CRUD ======================
# GET - posts 호출
@app.get('/posts', response_model=List[PostRead])
def get_posts():
	return posts
# GET - post 호출
@app.get('/post/{post_id}', response_model=PostRead)
def get_post(post_id: int):
	for post in posts:
		if post.id == post_id:
			return post
	raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# POST - post 추가
@app.post('/post', response_model=PostRead, status_code=status.HTTP_201_CREATED)
def create_post(new_post: PostCreate):
	'''게시글 생성'''
	global post_id_counter
	post_id_counter += 1
	post = PostRead(
		id = post_id_counter,
		email = new_post.email,
		title = new_post.title,
		content = new_post.content,
		likes = 0,
		views = 0,
		created_at = datetime.now(),
		updated_at = datetime.now(),
		comments=[]
	)
	posts.append(post)
	return post

# PATCH - post 수정
@app.patch('/posts/{post_id}', response_model=PostRead)
def update_post(post_id: int, update_post: PostUpdate):
	for post in posts:
		if post.id == post_id:
			if update_post.title is not None:
				post.title = update_post.title
			if update_post.content is not None:
				post.content = update_post.content
			post.updated_at = datetime.now()
			return post
	raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# DELETE - post 삭제
@app.delete('/posts/{post_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_post(post_id: int):
	global posts
	posts = [post for post in posts if post.id != post_id]
	global post_id_counter
	post_id_counter = 0
	return None


# ====================== Comment 엔드포인트 - 전체 CRUD ======================
# POST - comment 추가
@app.post('/posts/{post_id}/comments', response_model=CommentRead)
def create_comment(post_id: int, new_comment: CommentCreate):
	global comment_id_counter
	comment_id_counter += 1
	for post in posts:
		if post.id == post_id:
			comment = CommentRead(
				id = comment_id_counter,
				email = new_comment.email,
				content = new_comment.content,
				created_at = datetime.now()
			)
			post.comments.append(comment)
			return comment
	raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
# PATCH - comment 수정
@app.patch("/posts/{post_id}/comments/{comment_id}", response_model=CommentRead)
def update_comment(post_id: int, comment_id: int, update_comment: CommentUpdate):
	"""댓글 수정 (content만)"""
	for post in posts:
		if post.id == post_id:
			for comment in post.comments:
				if comment.id == comment_id:
					if update_comment.content is not None:
						comment.content = update_comment.content
					return comment
	raise HTTPException(status_code=404, detail="댓글을 찾을 수 없습니다.")
# DELETE - comment 삭제
@app.delete("/posts/{post_id}/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_comment(post_id: int, comment_id: int):
	"""댓글 삭제"""
	global comment_id_counter
	for post in posts:
		if post.id == post_id:
			post.comments = [comment for comment in post.comments if comment.id != comment_id]
			comment_id_counter = 0
			return None
	raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# ====================== 조회수 엔드포인트 (Update) ======================
# PATCH - views 변경
@app.patch("/posts/{post_id}/view")
def increment_view(post_id: int):
	for post in posts:
		if post.id == post_id:
			post.views += 1
			return {"post_id": post_id, "views": post.views}
	raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# ====================== 좋아요 엔드포인트 (Update) ======================
# PATCH - likes 변경
@app.patch("/posts/{post_id}/like")
def increment_like(post_id: int):
	for post in posts:
		if post.id == post_id:
			post.likes += 1
			return {"post_id": post_id, "likes": post.likes}
	raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")