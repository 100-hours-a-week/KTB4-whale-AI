from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from pydantic import EmailStr
from typing import List

class Post(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    email: EmailStr
    title: str
    content: str
    likes: int = Field(default=0)
    views: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    comments: List["Comment"] = Relationship(back_populates="post")

class Comment(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    email: EmailStr
    content: str
    created_at: datetime = Field(default_factory=datetime.now)

    post_id: int = Field(foreign_key="post.id")
    post: "Post" = Relationship(back_populates="comments")