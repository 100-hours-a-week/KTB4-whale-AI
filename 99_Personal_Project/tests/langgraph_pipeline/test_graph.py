"""
Test for langgraph_pipeline: graph.py

검증 항목:
1. build_rag_graph(): invoke() 결과가 RAGState 형태(question/retrieved/answer 키 보유)인가
2. build_rag_graph(): retrieved에 결과가 있을 때 generate 노드로 라우팅되어 answer가 채워지는가
3. build_rag_graph(): 빈 store일 때 no_results 노드로 라우팅되어 NO_RESULTS_ANSWER가 반환되는가
   (LCEL chain.py가 ValueError를 던졌던 경로가, LangGraph에서는 폴백 답변으로 처리된다)
4. build_rag_graph(): k가 retrieved 개수를 제한하는가
5. stream_rag_answer(): 토큰 리스트가 합쳐지면 전체 답변이 되는가
6. stream_rag_answer(): 빈 store일 때 NO_RESULTS_ANSWER를 단일 청크로 yield하는가
7. stream_rag_answer(): retrieve 노드 완료 후 llm.stream()을 쓰므로, FakeStreamingListLLM의
   토큰 단위 분할이 그대로 나오는가

[가짜 컴포넌트 선택 이유 — test_chain.py와 동일한 패턴]
- FakeListLLM: invoke() 경로 검증
- FakeStreamingListLLM: stream() 경로 검증
- RunnableLambda(lambda p: p): echo LLM — prompt 내용 검증
- DeterministicFakeEmbedding: 재현 가능한 임베딩
"""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM, FakeStreamingListLLM
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_pipeline.vector_store import build_vector_store
from langgraph_pipeline.graph import build_rag_graph, stream_rag_answer
from langgraph_pipeline.nodes import NO_RESULTS_ANSWER


@pytest.fixture
def fake_embedding():
    return DeterministicFakeEmbedding(size=8)


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="DaySync의 기본 API 포트는 9221입니다.", metadata={"idx": 0}),
        Document(page_content="일정 충돌 에러 코드는 SC-114입니다.", metadata={"idx": 1}),
        Document(page_content="DaySync의 내부 코드네임은 프로젝트 새벽별입니다.", metadata={"idx": 2}),
    ]


@pytest.fixture
def store(sample_documents, fake_embedding):
    return build_vector_store(sample_documents, fake_embedding)


@pytest.fixture
def empty_store(fake_embedding):
    return InMemoryVectorStore(embedding=fake_embedding)


@pytest.fixture
def initial_state():
    return {"question": "DaySync API 포트가 뭐야?", "retrieved": [], "answer": ""}


class TestBuildRagGraph:
    def test_invoke_result_has_rag_state_keys(self, store, initial_state):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        graph = build_rag_graph(store, llm, k=2)

        result = graph.invoke(initial_state)

        assert "question" in result
        assert "retrieved" in result
        assert "answer" in result

    def test_answer_is_filled_when_docs_retrieved(self, store, initial_state):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        graph = build_rag_graph(store, llm, k=2)

        result = graph.invoke(initial_state)

        assert result["answer"] == "가짜 답변입니다."

    def test_retrieved_contains_document_score_tuples(self, store, initial_state):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        graph = build_rag_graph(store, llm, k=2)

        result = graph.invoke(initial_state)

        for doc, score in result["retrieved"]:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_k_limits_retrieved_count(self, store, initial_state):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        graph = build_rag_graph(store, llm, k=2)

        result = graph.invoke(initial_state)

        assert len(result["retrieved"]) == 2

    def test_empty_store_routes_to_no_results_instead_of_raising(self, empty_store, initial_state):
        """
        LCEL build_rag_chain()은 빈 store에서 format_docs()의 ValueError를 전파했다.
        LangGraph 버전은 no_results 노드로 분기해 폴백 메시지를 반환한다.
        이것이 StateGraph 마이그레이션의 핵심 개선 사항 중 하나다.
        """
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        graph = build_rag_graph(empty_store, llm, k=3)

        result = graph.invoke(initial_state)

        assert result["answer"] == NO_RESULTS_ANSWER
        assert result["retrieved"] == []

    def test_empty_store_does_not_call_llm(self, empty_store, initial_state):
        """빈 store에서 no_results 분기가 올바르게 동작하면 llm.invoke()는 호출되지 않는다."""
        call_count = {"n": 0}

        def counting_llm(prompt):
            call_count["n"] += 1
            return "이 응답은 나오면 안 됩니다."

        graph = build_rag_graph(empty_store, RunnableLambda(counting_llm), k=3)
        graph.invoke(initial_state)

        assert call_count["n"] == 0

    def test_prompt_passed_to_llm_contains_question_and_context(self, store, initial_state):
        """echo LLM으로 그래프가 generate 노드에 조립해 넘기는 prompt 본문을 직접 검증한다."""
        echo_llm = RunnableLambda(lambda p: p)
        graph = build_rag_graph(store, echo_llm, k=2)

        result = graph.invoke(initial_state)
        prompt_text = result["answer"]

        assert f"질문: {initial_state['question']}" in prompt_text
        assert "[문서 1]" in prompt_text
        assert "Human:" not in prompt_text

    def test_question_is_preserved_in_result(self, store, initial_state):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        graph = build_rag_graph(store, llm, k=1)

        result = graph.invoke(initial_state)

        assert result["question"] == initial_state["question"]


class TestStreamRagAnswer:
    def test_tokens_join_to_full_answer(self, store):
        full_answer = "스트리밍 테스트 답변입니다"
        llm = FakeStreamingListLLM(responses=[full_answer])

        tokens = list(stream_rag_answer("DaySync API 포트가 뭐야?", store, llm, k=2))

        assert "".join(tokens) == full_answer

    def test_yields_multiple_chunks(self, store):
        """FakeStreamingListLLM은 응답을 글자 단위로 yield하므로 len > 1이어야 한다."""
        llm = FakeStreamingListLLM(responses=["스트리밍 답변"])

        tokens = list(stream_rag_answer("DaySync API 포트가 뭐야?", store, llm, k=2))

        assert len(tokens) > 1

    def test_empty_store_yields_no_results_answer(self, empty_store):
        llm = FakeListLLM(responses=["이 응답은 나오면 안 됩니다."])

        tokens = list(stream_rag_answer("아무 질문", empty_store, llm, k=3))

        assert tokens == [NO_RESULTS_ANSWER]

    def test_empty_store_does_not_call_llm_stream(self, empty_store):
        call_count = {"n": 0}

        def counting_stream(prompt):
            call_count["n"] += 1
            yield "이 토큰은 나오면 안 됩니다."

        class _CountingLLM:
            def stream(self, prompt):
                return counting_stream(prompt)

            def invoke(self, prompt):
                return "이 응답은 나오면 안 됩니다."

        list(stream_rag_answer("아무 질문", empty_store, _CountingLLM(), k=3))

        assert call_count["n"] == 0
