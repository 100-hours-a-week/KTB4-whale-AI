"""
langgraph_pipeline: 노드 함수 정의

StateGraph의 각 노드는 (RAGState) -> dict 시그니처를 가지는 함수다.
반환된 dict는 state에 병합(merge)된다 — 반환하지 않은 필드는 변경되지 않는다.

make_* 클로저 패턴:
    노드는 (state) -> dict 형태여야 하지만, store/llm/k 같은 실행 환경이 필요하다.
    이를 함수 인자로 받을 수 없으므로(StateGraph가 (state) 하나만 전달),
    클로저로 캡처하는 것이 StateGraph의 표준 패턴이다.

LCEL nodes.py가 없고 chain.py에서 바로 조립했던 것과 달리,
LangGraph에서는 노드(단계)와 그래프(조립)를 분리하는 것이 자연스럽다 —
노드를 독립적으로 테스트할 수 있고, 그래프 구조 변경 시 노드를 재사용할 수 있다.
"""
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_pipeline.prompt import format_docs, get_prompt_template
from langgraph_pipeline.state import RAGState

NO_RESULTS_ANSWER = "주어진 문서에서 답을 찾을 수 없습니다."


def make_retrieve_node(store: InMemoryVectorStore, k: int = 3):
    """
    retrieve 노드 팩토리.

    반환하는 노드 함수는 state["question"]으로 VectorStore를 검색하고,
    상위 k개 결과(Document, 유사도 점수) 튜플 리스트를 state["retrieved"]에 쓴다.

    검색 결과가 0개여도 예외를 던지지 않는다 — 빈 리스트를 그대로 반환해서
    _route_after_retrieve()가 "no_results" 분기로 보내게 한다.
    (기존 rag_pipeline.retriever.retrieve_top_k()는 빈 저장소에 ValueError를 던졌지만,
    LangGraph에서는 조건부 라우팅으로 처리하는 것이 더 자연스럽다.)
    """
    def retrieve_node(state: RAGState) -> dict:
        results = store.similarity_search_with_score(state["question"], k=k)
        return {"retrieved": results}

    return retrieve_node


def make_generate_node(llm: Runnable):
    """
    generate 노드 팩토리.

    반환하는 노드 함수는 state["retrieved"]의 문서들로 prompt를 조립하고
    llm.invoke()를 호출한 뒤, 결과를 state["answer"]에 쓴다.

    prompt 조립 로직(format_docs + get_prompt_template)은 langchain_pipeline.prompt에서
    그대로 가져온다 — LCEL chain.py의 _prompt_text_from_documents()와 동일한 절차.
    LangGraph 마이그레이션은 "조립 방식"을 바꾸는 것이지, prompt 포맷을 바꾸는 게 아니다.
    """
    def generate_node(state: RAGState) -> dict:
        documents = [doc for doc, _ in state["retrieved"]]
        context = format_docs(documents)
        prompt_value = get_prompt_template().invoke({"context": context, "question": state["question"]})
        prompt_text = prompt_value.to_messages()[0].content
        answer = llm.invoke(prompt_text)
        return {"answer": answer}

    return generate_node


def no_results_node(state: RAGState) -> dict:
    """
    no_results 노드 (조건부 라우팅의 폴백 경로).

    검색 결과가 없을 때 LLM을 호출하지 않고 고정 메시지를 반환한다.
    LCEL chain.py에는 없던 경로다 — LCEL은 빈 retrieved에서 format_docs()가
    ValueError를 던지며 실패했지만, StateGraph는 이 경우를 명시적인 노드로
    처리할 수 있다.
    """
    return {"answer": NO_RESULTS_ANSWER}
