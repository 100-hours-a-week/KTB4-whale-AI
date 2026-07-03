"""
Test for langgraph_pipeline: nodes.py

검증 항목:
1. make_retrieve_node: 질문으로 store를 검색해 {"retrieved": [...]} 형태로 반환하는가
2. make_retrieve_node: 빈 store일 때 예외 없이 {"retrieved": []}를 반환하는가
   (기존 rag_pipeline.retriever의 ValueError와 달리, 노드는 예외 대신 빈 리스트를 반환한다)
3. make_retrieve_node: k가 검색 결과 개수를 제한하는가
4. make_generate_node: retrieved로 prompt를 조립해 llm.invoke()를 호출하고 {"answer": str}을 반환하는가
5. make_generate_node: prompt에 "[문서 N]"과 질문이 포함되어 있는가 (echo LLM으로 검증)
6. no_results_node: retrieved 없이도 고정 메시지를 {"answer": ...}로 반환하는가

[가짜 컴포넌트 선택 이유]
- DeterministicFakeEmbedding: 재현 가능한 벡터 생성 (4~8단계 테스트와 동일한 패턴)
- FakeListLLM: invoke() 결과 검증용
- RunnableLambda(lambda p: p): echo LLM — prompt 본문 자체를 answer로 받아 내용 검증
"""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_pipeline.vector_store import build_vector_store
from langgraph_pipeline.nodes import NO_RESULTS_ANSWER, make_generate_node, make_retrieve_node, no_results_node
from langgraph_pipeline.state import RAGState


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
def base_state() -> RAGState:
    return {"question": "DaySync API 포트가 뭐야?", "retrieved": [], "answer": ""}


class TestMakeRetrieveNode:
    def test_returns_retrieved_key_with_list(self, store, base_state):
        node = make_retrieve_node(store, k=2)
        result = node(base_state)
        assert "retrieved" in result
        assert isinstance(result["retrieved"], list)

    def test_each_item_is_document_score_tuple(self, store, base_state):
        node = make_retrieve_node(store, k=2)
        result = node(base_state)
        for item in result["retrieved"]:
            doc, score = item
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_k_limits_result_count(self, store, base_state):
        node = make_retrieve_node(store, k=2)
        result = node(base_state)
        assert len(result["retrieved"]) == 2

    def test_empty_store_returns_empty_list_without_exception(self, fake_embedding, base_state):
        empty_store = InMemoryVectorStore(embedding=fake_embedding)
        node = make_retrieve_node(empty_store, k=3)
        result = node(base_state)
        assert result["retrieved"] == []

    def test_does_not_overwrite_other_state_keys(self, store, base_state):
        """노드는 담당 필드("retrieved")만 반환해야 한다 — 다른 키를 덮어쓰지 않는다."""
        node = make_retrieve_node(store, k=1)
        result = node(base_state)
        assert set(result.keys()) == {"retrieved"}


class TestMakeGenerateNode:
    @pytest.fixture
    def state_with_retrieved(self, store) -> RAGState:
        node = make_retrieve_node(store, k=2)
        retrieved = node({"question": "DaySync API 포트가 뭐야?", "retrieved": [], "answer": ""})["retrieved"]
        return {"question": "DaySync API 포트가 뭐야?", "retrieved": retrieved, "answer": ""}

    def test_returns_answer_key(self, state_with_retrieved):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        node = make_generate_node(llm)
        result = node(state_with_retrieved)
        assert "answer" in result

    def test_answer_matches_llm_response(self, state_with_retrieved):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        node = make_generate_node(llm)
        result = node(state_with_retrieved)
        assert result["answer"] == "가짜 답변입니다."

    def test_prompt_contains_question_and_context_blocks(self, state_with_retrieved):
        """echo LLM으로 노드가 조립한 prompt 본문을 직접 검증한다."""
        echo_llm = RunnableLambda(lambda p: p)
        node = make_generate_node(echo_llm)
        result = node(state_with_retrieved)
        prompt_text = result["answer"]

        assert f"질문: {state_with_retrieved['question']}" in prompt_text
        assert "답변:" in prompt_text
        assert "[문서 1]" in prompt_text
        assert "Human:" not in prompt_text

    def test_does_not_overwrite_other_state_keys(self, state_with_retrieved):
        llm = FakeListLLM(responses=["가짜 답변입니다."])
        node = make_generate_node(llm)
        result = node(state_with_retrieved)
        assert set(result.keys()) == {"answer"}


class TestNoResultsNode:
    def test_returns_answer_key(self):
        state: RAGState = {"question": "아무 질문", "retrieved": [], "answer": ""}
        result = no_results_node(state)
        assert "answer" in result

    def test_answer_is_no_results_constant(self):
        state: RAGState = {"question": "아무 질문", "retrieved": [], "answer": ""}
        result = no_results_node(state)
        assert result["answer"] == NO_RESULTS_ANSWER

    def test_does_not_overwrite_other_state_keys(self):
        state: RAGState = {"question": "아무 질문", "retrieved": [], "answer": ""}
        result = no_results_node(state)
        assert set(result.keys()) == {"answer"}
