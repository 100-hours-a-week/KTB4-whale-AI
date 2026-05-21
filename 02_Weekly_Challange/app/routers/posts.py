from app.schemas import PostRead, PostCreate, PostUpdate
from app.database import get_session
from app.crud.post import get_posts, get_post, create_post, update_post, delete_post, increment_view, increment_like
from fastapi import APIRouter, Depends, status, HTTPException
from sqlmodel import Session
from typing import List

router = APIRouter(prefix="/posts", tags=["posts"])

# ====================== 게시글(Post) 엔드포인트 ======================
@router.get('/', response_model=List[PostRead])
def get_all_posts(session: Session = Depends(get_session)):
    return get_posts(session)

@router.get('/{post_id}', response_model=PostRead)
def get_single_post(post_id: int, session: Session = Depends(get_session)):
    post = get_post(post_id, session)
    if post:
        return post
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@router.post('/', response_model=PostRead, status_code=status.HTTP_201_CREATED)
def create_new_post(new_post: PostCreate, session: Session = Depends(get_session)):
    return create_post(new_post, session)

@router.patch('/{post_id}', response_model=PostRead)
def update_existing_post(post_id: int, update_data: PostUpdate, session: Session = Depends(get_session)):
    post = update_post(post_id, update_data, session)
    if post:
        return post
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@router.delete('/{post_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_existing_post(post_id: int, session: Session = Depends(get_session)):
    post = delete_post(post_id, session)
    if post:
       pass 
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# ====================== 조회수 & 좋아요 ======================
@router.patch("/{post_id}/view")
def increment_existing_view(post_id: int, session: Session = Depends(get_session)):
    views = increment_view(post_id, session)
    if views:
        return {"post_id": post_id, "views": views}
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

@router.patch("/{post_id}/like")
def increment_existing_like(post_id: int, session: Session = Depends(get_session)):
    likes = increment_like(post_id, session)
    if likes:
        return {"post_id": post_id, "likes": likes}
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")