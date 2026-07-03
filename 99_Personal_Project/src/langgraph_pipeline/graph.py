"""
langgraph_pipeline: StateGraph 조립

LCEL chain.py와의 핵심 차이:
1. 상태(State)가 명시적 — RAGState TypedDict로 모든 단계가 공유하는 데이터를 선언한다.
2. 조건부 라우팅 — 검색 결과가 없으면 LLM을 호출하지 않고 no_results 노드로 분기한다.
   LCEL의 | 연산자로는 표현할 수 없는 제어 흐름이다.
3. 노드 단위 가시성 — graph.stream(stream_mode="updates")로 각 노드의 출력을 따로 볼 수 있다.

그래프 구조:
    START
      ↓
    [retrieve]  state["retrieved"]를 채운다
      ↓ (_route_after_retrieve 조건부)
      ├── retrieved 비어 있음 → [no_results] → END
      └── retrieved 있음     → [generate]   → END

invoke()와 stream() 경로:
  - build_rag_graph().invoke()   : 전체 그래프를 실행하고 최종 RAGState를 반환 (/query 대응)
  - stream_rag_answer()          : retrieve 완료 즉시 llm.stream()으로 토큰을 yield (/query/stream 대응)

    stream_rag_answer()가 StateGraph.stream()을 쓰면서도 llm.stream()을 직접 호출하는 이유:
    LangGraph의 stream_mode="messages"는 ChatModel(AIMessage를 반환하는 모델)에서만
    토큰 단위 스트리밍을 지원한다. 이 프로젝트의 LLM은 BaseLLM(HuggingFacePipeline,
    CustomTransformerLLM)이므로, stream_mode="messages"로는 토큰이 나오지 않는다.
    대신 graph.stream(stream_mode="updates")로 retrieve 노드의 완료 시점을 포착한 뒤,
    llm.stream()으로 직접 토큰을 yield하는 혼합 방식을 쓴다.
"""
from __future__ import annotations

from typing import Iterator

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

from langchain_pipeline.prompt import format_docs, get_prompt_template
from langgraph_pipeline.nodes import NO_RESULTS_ANSWER, make_generate_node, make_retrieve_node, no_results_node
from langgraph_pipeline.state import RAGState


def _route_after_retrieve(state: RAGState) -> str:
    """retrieve 노드 이후 라우팅 조건.

    state["retrieved"]가 비어 있으면 "no_results", 있으면 "generate"를 반환한다.
    add_conditional_edges()의 두 번째 인자로 쓰인다.
    """
    return "generate" if state["retrieved"] else "no_results"


def build_rag_graph(
    store: InMemoryVectorStore,
    llm: Runnable,
    k: int = 3,
):
    """
    기존 langchain_pipeline.chain.build_rag_chain()에 대응하는 CompiledStateGraph를 반환한다.

    Args:
        store: build_vector_store()로 만든 InMemoryVectorStore.
        llm: get_gemma_llm() 또는 get_custom_transformer_llm()의 반환값 (둘 다 Runnable).
        k: 검색할 상위 문서 개수.

    Returns:
        invoke({"question": str, "retrieved": [], "answer": ""}) ->
            RAGState {"question": str, "retrieved": [...], "answer": str}

        LCEL build_rag_chain()과의 반환 형태 차이:
          - LCEL : {"answer": str, "retrieved_chunks": [{"text", "score"}, ...]}
          - 이 그래프 : RAGState (state["retrieved"]에 (Document, float) 튜플이 그대로 담김)
        호출부(main.py)에서 state["retrieved"]를 순회해 RetrievedChunk로 변환한다.
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", make_retrieve_node(store, k))
    graph.add_node("generate", make_generate_node(llm))
    graph.add_node("no_results", no_results_node)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate": "generate", "no_results": "no_results"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("no_results", END)

    return graph.compile()


def stream_rag_answer(
    question: str,
    store: InMemoryVectorStore,
    llm: Runnable,
    k: int = 3,
) -> Iterator[str]:
    """
    기존 langchain_pipeline.chain.build_answer_only_chain().stream()에 대응하는 스트리밍 생성기.

    graph.stream(stream_mode="updates")로 retrieve 노드가 완료되는 시점을 포착하고,
    이후 llm.stream()으로 토큰을 직접 yield한다.
    검색 결과가 없으면 NO_RESULTS_ANSWER를 단일 청크로 yield하고 종료한다.

    stream_mode="updates"는 {node_name: state_update} 딕셔너리를 노드 완료마다 yield한다.
    retrieve 노드의 update를 받은 즉시 break하므로, generate 노드는 실행되지 않는다
    (llm 호출이 두 번 일어나지 않는다).

    Args:
        question: 사용자 질문.
        store: build_vector_store()로 만든 InMemoryVectorStore.
        llm: get_gemma_llm() 또는 get_custom_transformer_llm()의 반환값.
        k: 검색할 상위 문서 개수.

    Yields:
        str: 생성된 텍스트 조각. SSE "data: " 래핑은 호출부(main.py)에서 처리한다.
    """
    initial_state: RAGState = {"question": question, "retrieved": [], "answer": ""}
    graph = build_rag_graph(store, llm, k)

    retrieved = []
    for update in graph.stream(initial_state, stream_mode="updates"):
        if "retrieve" in update:
            retrieved = update["retrieve"].get("retrieved", [])
            break  # retrieve 완료 즉시 스트리밍 시작 — generate 노드는 실행하지 않는다

    if not retrieved:
        yield NO_RESULTS_ANSWER
        return

    documents = [doc for doc, _ in retrieved]
    context = format_docs(documents)
    prompt_value = get_prompt_template().invoke({"context": context, "question": question})
    prompt_text = prompt_value.to_messages()[0].content

    for token in llm.stream(prompt_text):
        yield token


if __name__ == "__main__":
    from paths import DATA_DIR
    from langchain_pipeline.embedding import get_embeddings_model
    from langchain_pipeline.llm import get_gemma_llm
    from langchain_pipeline.loader import load_document
    from langchain_pipeline.splitter import split_fixed_size
    from langchain_pipeline.vector_store import build_vector_store

    sample_path = DATA_DIR / "daysync_manual.md"
    documents = load_document(str(sample_path))
    chunks = split_fixed_size(documents, chunk_size=300, chunk_overlap=50)

    embeddings_model = get_embeddings_model()
    store = build_vector_store(chunks, embeddings_model)

    print("[Gemma 4 E2B-it 로딩 중... 처음 실행 시 다운로드가 필요합니다]")
    llm = get_gemma_llm()

    question = "DaySync의 내부 코드네임은 무엇인가?"

    # /query 대응
    graph = build_rag_graph(store, llm, k=3)
    result = graph.invoke({"question": question, "retrieved": [], "answer": ""})
    print(f"[질문] {question}\n")
    print(f"[답변] {result['answer']}\n")
    print("[검색된 chunk]")
    for doc, score in result["retrieved"]:
        print(f"  (score={score:.4f}) {doc.page_content[:80]!r}")

    # /query/stream 대응
    print("\n[스트리밍 답변] ", end="", flush=True)
    for token in stream_rag_answer(question, store, llm, k=3):
        print(token, end="", flush=True)
    print()
