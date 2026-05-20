from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional
from ollama import chat

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

# Summary
class SummaryRead(BaseModel):
	summary: str = "요약 생성 불가"

# ====================== In-memory 저장소 ======================
posts: List[PostRead] = []
post_id_counter = 0
comment_id_counter = 0
summarys: List[SummaryRead] = []

# ====================== Post 엔드포인트 (전체 CRUD) ======================
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


# ====================== Comment 엔드포인트 (Create, Update, Delete) ======================
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

# ====================== AI 요약 엔드포인트 (Create) ======================
# POST - 게시글 요약
@app.post('/posts/summary', response_model=SummaryRead)
def create_posts_summary():
	"""1. 전체 게시글 목록 요약"""
	# if not posts:
	# 	return SummaryRead(summary="아직 게시글이 없습니다.")
	
	# TODO: 더미데이터 지우기
	dummy_posts: List[PostRead] = [
		PostRead(
			id=1,
			email="test1@example.com",
			title="FastAPI로 REST API 만들기",
			content="FastAPI와 Pydantic을 활용한 백엔드 개발 방법을 공유합니다.",
			likes=0,
			views=0,
			created_at=datetime.now(),
			updated_at=datetime.now(),
			comments= [
				CommentRead(
					id=1,
					email="user1@example.com",
					content="FastAPI 정말 편리하네요! 자동 문서화가 최고예요.",
					created_at=datetime(2026, 5, 20, 13, 0, 0)
				),
				CommentRead(
					id=2,
					email="user2@example.com",
					content="Ollama 연동도 잘 되네요. AI 요약 기능 기대돼요!",
					created_at=datetime(2026, 5, 20, 13, 5, 0)
				),
				CommentRead(
					id=3,
					email="user3@example.com",
					content="gemma4:e4b 모델로 요약 속도가 빠른 편인가요?",
					created_at=datetime(2026, 5, 20, 13, 10, 0)
				)
			]
		),
		PostRead(
			id=2,
			email="test2@example.com",
			title="Ollama gemma4:e4b 사용 후기",
			content="로컬 LLM을 FastAPI에 연동해서 게시글 요약 기능을 구현해보았습니다.",
			likes=0,
			views=0,
			created_at=datetime.now(),
			updated_at=datetime.now(),
			comments= [
				CommentRead(
					id=1,
					email="user1@example.com",
					content="FastAPI 정말 편리하네요! 자동 문서화가 최고예요.",
					created_at=datetime(2026, 5, 20, 13, 0, 0)
				),
				CommentRead(
					id=2,
					email="user2@example.com",
					content="Ollama 연동도 잘 되네요. AI 요약 기능 기대돼요!",
					created_at=datetime(2026, 5, 20, 13, 5, 0)
				),
				CommentRead(
					id=3,
					email="user3@example.com",
					content="gemma4:e4b 모델로 요약 속도가 빠른 편인가요?",
					created_at=datetime(2026, 5, 20, 13, 10, 0)
				)
			]
		),
		PostRead(
			id=3,
			email="test3@example.com",
			title="Python snake_case vs camelCase",
			content="FastAPI 프로젝트에서 필드명 컨벤션에 대해 이야기해봅니다.",
			likes=0,
			views=0,
			created_at=datetime.now(),
			updated_at=datetime.now(),
			comments= [
				CommentRead(
					id=1,
					email="user1@example.com",
					content="FastAPI 정말 편리하네요! 자동 문서화가 최고예요.",
					created_at=datetime(2026, 5, 20, 13, 0, 0)
				),
				CommentRead(
					id=2,
					email="user2@example.com",
					content="Ollama 연동도 잘 되네요. AI 요약 기능 기대돼요!",
					created_at=datetime(2026, 5, 20, 13, 5, 0)
				),
				CommentRead(
					id=3,
					email="user3@example.com",
					content="gemma4:e4b 모델로 요약 속도가 빠른 편인가요?",
					created_at=datetime(2026, 5, 20, 13, 10, 0)
				)
			]
		),
		PostRead(
			id=4,
			email="test4@example.com",
			title="커뮤니티 사이트 기획 및 개발 과정",
			content="기획 → 개발 → 배포까지의 전체 과정을 정리했습니다.",
			likes=0,
			views=0,
			created_at=datetime.now(),
			updated_at=datetime.now(),
			comments= [
				CommentRead(
					id=1,
					email="user1@example.com",
					content="FastAPI 정말 편리하네요! 자동 문서화가 최고예요.",
					created_at=datetime(2026, 5, 20, 13, 0, 0)
				),
				CommentRead(
					id=2,
					email="user2@example.com",
					content="Ollama 연동도 잘 되네요. AI 요약 기능 기대돼요!",
					created_at=datetime(2026, 5, 20, 13, 5, 0)
				),
				CommentRead(
					id=3,
					email="user3@example.com",
					content="gemma4:e4b 모델로 요약 속도가 빠른 편인가요?",
					created_at=datetime(2026, 5, 20, 13, 10, 0)
				)
			]
		),
	]
	
	post_text = '\n\n'.join(
        [f"title:{post.title}\ncontent:{post.content}" for post in dummy_posts]
    )

	prompt = f"""
				아래 게시글을 종합해서 한국어로 자연스럽게 요약해주세요.
				전체적인 주제, 주요 의견, 분위기를 200자 이내로 간결하게 작성하라.

				{post_text}
			  """
	
	try:
		response = chat(
			model='gemma4:e4b',
			messages=[{'role': 'user', 'content': prompt}]
		)
		summarized_posts = response.message.content
		print(summarized_posts)
		return SummaryRead(summary=summarized_posts) if summarized_posts is not None else SummaryRead()
	except Exception as e:
		raise HTTPException(
			status_code=500,
			detail=f"Ollama 호출 실패: {str(e)} (Ollama 서버가 실행 중인지 확인하세요)"
		)

# POST - 댓글 요약
@app.post('/posts/{post_id}/comments/summary', response_model=SummaryRead)
def create_comments_summary(post_id: int):
	"""현재 게시글의 댓글 목록 요약"""
	# TODO: 더미데이터 지우기
	dummy_posts: List[PostRead] = [
		PostRead(
			id=1,
			email="test1@example.com",
			title="FastAPI로 REST API 만들기",
			content="FastAPI와 Pydantic을 활용한 백엔드 개발 방법을 공유합니다.",
			likes=0,
			views=0,
			created_at=datetime.now(),
			updated_at=datetime.now(),
			comments= [
				CommentRead(
					id=1,
					email="user1@example.com",
					content="FastAPI 정말 편리하네요! 자동 문서화가 최고예요.",
					created_at=datetime(2026, 5, 20, 13, 0, 0)
				),
				CommentRead(
					id=2,
					email="user2@example.com",
					content="Ollama 연동도 잘 되네요. AI 요약 기능 기대돼요!",
					created_at=datetime(2026, 5, 20, 13, 5, 0)
				),
				CommentRead(
					id=3,
					email="user3@example.com",
					content="gemma4:e4b 모델로 요약 속도가 빠른 편인가요?",
					created_at=datetime(2026, 5, 20, 13, 10, 0)
				)
			]
		),
		PostRead(
			id=2,
			email="test2@example.com",
			title="Ollama gemma4:e4b 사용 후기",
			content="로컬 LLM을 FastAPI에 연동해서 게시글 요약 기능을 구현해보았습니다.",
			likes=0,
			views=0,
			created_at=datetime.now(),
			updated_at=datetime.now(),
			comments= [
				CommentRead(
					id=1,
					email="user1@example.com",
					content="FastAPI 정말 편리하네요! 자동 문서화가 최고예요.",
					created_at=datetime(2026, 5, 20, 13, 0, 0)
				),
				CommentRead(
					id=2,
					email="user2@example.com",
					content="Ollama 연동도 잘 되네요. AI 요약 기능 기대돼요!",
					created_at=datetime(2026, 5, 20, 13, 5, 0)
				),
				CommentRead(
					id=3,
					email="user3@example.com",
					content="gemma4:e4b 모델로 요약 속도가 빠른 편인가요?",
					created_at=datetime(2026, 5, 20, 13, 10, 0)
				)
			]
		),
		PostRead(
			id=3,
			email="test3@example.com",
			title="Python snake_case vs camelCase",
			content="FastAPI 프로젝트에서 필드명 컨벤션에 대해 이야기해봅니다.",
			likes=0,
			views=0,
			created_at=datetime.now(),
			updated_at=datetime.now(),
			comments= [
				CommentRead(
					id=1,
					email="user1@example.com",
					content="FastAPI 정말 편리하네요! 자동 문서화가 최고예요.",
					created_at=datetime(2026, 5, 20, 13, 0, 0)
				),
				CommentRead(
					id=2,
					email="user2@example.com",
					content="Ollama 연동도 잘 되네요. AI 요약 기능 기대돼요!",
					created_at=datetime(2026, 5, 20, 13, 5, 0)
				),
				CommentRead(
					id=3,
					email="user3@example.com",
					content="gemma4:e4b 모델로 요약 속도가 빠른 편인가요?",
					created_at=datetime(2026, 5, 20, 13, 10, 0)
				)
			]
		),
		PostRead(
			id=4,
			email="test4@example.com",
			title="커뮤니티 사이트 기획 및 개발 과정",
			content="기획 → 개발 → 배포까지의 전체 과정을 정리했습니다.",
			likes=0,
			views=0,
			created_at=datetime.now(),
			updated_at=datetime.now(),
			comments= [
				CommentRead(
					id=1,
					email="user1@example.com",
					content="FastAPI 정말 편리하네요! 자동 문서화가 최고예요.",
					created_at=datetime(2026, 5, 20, 13, 0, 0)
				),
				CommentRead(
					id=2,
					email="user2@example.com",
					content="Ollama 연동도 잘 되네요. AI 요약 기능 기대돼요!",
					created_at=datetime(2026, 5, 20, 13, 5, 0)
				),
				CommentRead(
					id=3,
					email="user3@example.com",
					content="gemma4:e4b 모델로 요약 속도가 빠른 편인가요?",
					created_at=datetime(2026, 5, 20, 13, 10, 0)
				)
			]
		),
	]
	
	for post in dummy_posts: 
		if post.id == post_id:
			if not post.comments:
				return SummaryRead(summary="아직 댓글이 없습니다.")
			
			comments_text = '\n\n'.join(
				[f"내용: {comment.content}" for comment in post.comments]
			)

			prompt = f"""
						아래 댓글들을 종합해서 한국어로 자연스럽게 요약해주세요.
						전체적인 주제, 주요 의견, 분위기를 200자 이내로 간결하게 작성해주세요.

						댓글 목록:
							{comments_text}
					  """
			
			try:
				response = chat(
					model='gemma4:e4b',
					messages=[{'role': 'user', 'content': prompt}]
				)

				summarized_comments = response.message.content
				print(summarized_comments)
				
				return SummaryRead(summary=summarized_comments) if summarized_comments is not None else SummaryRead()
			except Exception as e:
				raise HTTPException(
					status_code=500,
					detail="Ollama 호출 실패: {str(e)} (Ollama 서버가 실행 중인지 확인하세요)"
				)
		break

