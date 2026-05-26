from app.models import Post
from app.schemas import PostCreate, PostUpdate
from sqlmodel import Session, select
from datetime import datetime

# ====================== 게시글(Post) 엔드포인트 ======================
def get_posts(session: Session):
    """모든 게시글 목록"""
    return session.exec(select(Post)).all()

def get_post(post_id: int, session: Session):
    """특정 게시글 조회 (없으면 None 반환)"""
    return session.get(Post, post_id)

def create_post(new_post: PostCreate, session: Session):
    """게시글 생성"""
    post = Post.model_validate(new_post)
    session.add(post)
    session.commit()
    session.refresh(post)
    return post

def update_post(post_id: int, update_data: PostUpdate, session: Session):
    """게시글 수정 (없으면 None 반환)"""
    post = session.get(Post, post_id)

    if post:
        post.title = update_data.title if update_data.title is not None else post.title
        post.content = update_data.content if update_data.content is not None else post.content
        post.updated_at = datetime.now()
        session.add(post)
        session.commit()
        session.refresh(post)
        return post

def delete_post(post_id: int, session: Session):
    """게시글 삭제 (성공하면 True, 실패하면 False)"""
    post = session.get(Post, post_id)

    if post:
        session.delete(post)
        session.commit()
        return True
    return False

# ====================== 조회수 & 좋아요 ======================
def increment_view(post_id: int, session: Session):
    post = session.get(Post, post_id)
    if post:
        post.views += 1
        session.add(post)
        session.commit()
        session.refresh(post)
        return post.views
    return None

def increment_like(post_id: int, session: Session):
    post = session.get(Post, post_id)
    if post:
        post.likes += 1
        session.add(post)
        session.commit()
        session.refresh(post)
        return post.likes
    return None