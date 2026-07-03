"""
langgraph_pipeline: 공유 상태 정의

StateGraph의 모든 노드가 읽고 쓰는 공유 데이터 구조.
LCEL chain.py는 상태가 암묵적(Runnable 간 파이프로 전달)이었지만,
StateGraph는 상태를 TypedDict로 명시적으로 선언한다.

각 필드의 업데이트 방식: reducer 없음 → 노드의 반환값이 해당 필드를 덮어쓴다.
(LangGraph 기본 동작 — add_messages처럼 누적하지 않고 overwrite)

필드:
    question: 사용자의 원본 질문. START에서 주입되며 이후 노드에서 읽기 전용으로 사용된다.
    retrieved: retrieve 노드가 채우는 (Document, 유사도 점수) 튜플 리스트.
               초기값: [] — 조건부 라우팅(_route_after_retrieve)이 빈 리스트를 "결과 없음"으로 판단한다.
    answer: generate 또는 no_results 노드가 채우는 최종 답변 문자열.
            초기값: "" — END 도달 전에 반드시 어느 한 노드가 채운다.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    retrieved: list[tuple[Document, float]]
    answer: str
