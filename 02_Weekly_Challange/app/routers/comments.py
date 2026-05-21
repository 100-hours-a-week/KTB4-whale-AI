from app.schemas import CommentRead, CommentCreate, CommentUpdate
from app.database import get_session
from app.crud.comment import create_comment, update_comment, delete_comment
from fastapi import APIRouter, Depends, status, HTTPException
from sqlmodel import Session

router = APIRouter(prefix="/posts", tags=["posts"])

# ====================== 댓글(Comment) 엔드포인트 ======================
@router.post('/{post_id}/comments', response_model=CommentRead, status_code=status.HTTP_201_CREATED)
def create_new_comment(post_id: int, new_comment: CommentCreate, session: Session = Depends(get_session)):
    comment = create_comment(post_id, new_comment, session)
    if comment:
        return comment
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
    

@router.patch("/{post_id}/comments/{comment_id}", response_model=CommentRead)
def update_existing_comment(post_id: int, comment_id: int, update_data: CommentUpdate, session: Session = Depends(get_session)):
    comment = update_comment(post_id, comment_id, update_data, session)
    if comment:
        return comment
    raise HTTPException(status_code=404, detail="댓글을 찾을 수 없습니다.")

@router.delete("/{post_id}/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_existing_comment(post_id: int, comment_id: int, session: Session = Depends(get_session)):
    comment = delete_comment(post_id, comment_id, session)
    if comment:
        pass
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")