from __future__ import annotations
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional

# ====================== Post Schemas ======================
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
    comments: List["CommentRead"] = []

# ====================== Comment Schemas ======================
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

# ====================== Summary Schema ======================
class SummaryRead(BaseModel):
    summary: str = "요약 생성 불가"