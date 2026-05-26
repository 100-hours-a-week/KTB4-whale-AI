from app.models import Post, Comment
from app.schemas import CommentCreate, CommentUpdate
from sqlmodel import Session

def create_comment(post_id: int, new_comment: CommentCreate, session: Session):
    post = session.get(Post, post_id)
    if post:
        comment = Comment.model_validate(new_comment)
        comment.post_id = post_id
        session.add(comment)
        session.commit()
        session.refresh(comment)
        return comment
    return None
    
def update_comment(post_id: int, comment_id: int, update_data: CommentUpdate, session: Session):
    comment = session.get(Comment, comment_id)
    if comment and comment.post_id == post_id:
        comment.content = update_data.content if update_data.content is not None else comment.content
        session.add(comment)
        session.commit()
        session.refresh(comment)
        return comment
    return None

def delete_comment(post_id: int, comment_id: int, session: Session):
    comment = session.get(Comment, comment_id)
    if comment and comment.post_id == post_id:
        session.delete(comment)
        session.commit()
        return True
    return False
    