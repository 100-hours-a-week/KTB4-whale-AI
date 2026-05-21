from app.models import Post
from app.schemas import SummaryRead
from app.database import get_session
from app.crud.post import get_posts, get_post
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from ollama import chat

router = APIRouter(prefix="/posts", tags=["posts"])

# ====================== AI 요약 엔드포인트 ======================
@router.post('/summary', response_model=SummaryRead)
def create_posts_summary(session: Session = Depends(get_session)):
    """1. 전체 게시글 목록 요약"""
    posts = get_posts(session)
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

@router.post('/{post_id}/comments/summary', response_model=SummaryRead)
def create_comments_summary(post_id: int, session: Session = Depends(get_session)):
    """2. 특정 게시글의 댓글 목록 요약"""
    post = get_post(post_id, session)
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