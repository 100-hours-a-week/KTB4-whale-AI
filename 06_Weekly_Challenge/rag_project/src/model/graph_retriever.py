"""
Step C-4: Graph RAG — Graph Retrieval (키워드 매칭 기반)

책임(Responsibility): 사용자 질문과 그래프를 받아, 관련된 노드와 그 노드에
연결된 엣지(관계)를 찾아 반환한다. Prompt 조립이나 생성은 다음 단계의 책임.

탐색 전략 ("근본에서 확장" 원칙 — 가장 단순한 방법부터 시작):
1. 질문 문자열에 그래프 노드 ID가 그대로 포함되어 있는지 확인하여 시작 노드를 찾는다.
2. 시작 노드와 연결된 모든 엣지(양방향: source 또는 target으로 등장하는 엣지)를 수집한다.
3. 여러 시작 노드가 발견되면, 각각의 연결된 엣지를 모두 합친다.

한계: 질문에 그래프 노드 이름이 정확히 등장하지 않으면 시작 노드를 찾지 못한다.
이 경우 더 발전된 전략(LLM에게 탐색 계획을 추론시키는 방식)으로 확장이 필요하다.
"""


def find_start_nodes(question: str, graph: dict) -> list[str]:
    """
    질문 문자열에 포함된 그래프 노드 ID를 찾아 반환한다.

    Args:
        question: 사용자 질문
        graph: {"nodes": [...], "edges": [...]} 형태의 그래프

    Returns:
        질문에 등장한 노드 ID 리스트
    """
    question_lower = question.lower()
    matched_nodes = []

    for node in graph["nodes"]:
        node_id = node["id"]
        if node_id.lower() in question_lower:
            matched_nodes.append(node_id)

    return matched_nodes


def retrieve_related_edges(question: str, graph: dict, max_hops: int = 2) -> list[dict]:
    """
    질문과 관련된 노드를 찾고, 그 노드로부터 최대 max_hops만큼 떨어진
    모든 엣지를 너비 우선 탐색(BFS)으로 수집한다.

    예: "NF-227을 겪은 팀의 담당자는?" 같은 질문은, 1-hop만으로는
    "Team Falcon -- experienced_error --> NF-227"만 찾고, 담당자(Mina Park)로
    이어지는 "Team Falcon -- managed_by --> Mina Park"는 놓치게 된다.
    max_hops=2로 설정하면, 1-hop에서 새로 발견된 노드(Team Falcon)를 다음
    탐색의 시작점으로 추가하여, 그 노드와 연결된 엣지까지 마저 수집한다.

    Args:
        question: 사용자 질문
        graph: {"nodes": [...], "edges": [...]} 형태의 그래프
        max_hops: 시작 노드로부터 탐색할 최대 깊이. 기본값 2.

    Returns:
        관련된 엣지(dict)의 리스트. 각 엣지는 {"source", "relation", "target"} 형태.
        같은 엣지가 중복으로 포함되지 않는다.

    Raises:
        ValueError: 질문에서 시작 노드를 하나도 찾지 못했을 경우
    """
    start_nodes = find_start_nodes(question, graph)

    if not start_nodes:
        raise ValueError(
            f"질문 '{question}'에서 그래프 노드를 찾지 못했습니다. "
            "키워드 매칭 전략의 한계입니다 — 질문에 노드 이름이 정확히 등장해야 합니다."
        )

    visited_nodes: set[str] = set(start_nodes)
    current_frontier: set[str] = set(start_nodes)
    collected_edges: list[dict] = []
    seen_edge_keys: set[tuple[str, str, str]] = set()

    for _ in range(max_hops):
        next_frontier: set[str] = set()

        for edge in graph["edges"]:
            touches_frontier = edge["source"] in current_frontier or edge["target"] in current_frontier
            if not touches_frontier:
                continue

            edge_key = (edge["source"], edge["relation"], edge["target"])
            if edge_key not in seen_edge_keys:
                seen_edge_keys.add(edge_key)
                collected_edges.append(edge)

            # 아직 방문하지 않은 새 노드는 다음 hop의 탐색 시작점이 된다.
            for node_id in (edge["source"], edge["target"]):
                if node_id not in visited_nodes:
                    next_frontier.add(node_id)
                    visited_nodes.add(node_id)

        if not next_frontier:
            break  # 더 이상 확장할 새 노드가 없으면 조기 종료

        current_frontier = next_frontier

    return collected_edges


def edges_to_context(edges: list[dict]) -> str:
    """
    엣지 리스트를 LLM에게 전달할 텍스트 형태로 변환한다.

    Args:
        edges: retrieve_related_edges()의 반환값

    Returns:
        "A -- relation --> B" 형태의 줄들로 구성된 문자열
    """
    lines = [f"{edge['source']} --{edge['relation']}--> {edge['target']}" for edge in edges]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from model.document_loader import load_document
    from model.generator import TextGenerator
    from model.graph_extractor import build_graph, extract_relations

    sample_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "nimbusflow_team_incidents.md"
    )
    document = load_document(str(sample_path))
    summary_section = document.split("## 4. Summary Table")[1]

    print("[Gemma 4 E2B-it 로딩 중...]")
    generator = TextGenerator()

    relations = extract_relations(summary_section, generator)
    graph = build_graph(relations)

    questions = [
        "Who is the manager of the team that experienced NF-227?",
        "What configuration change did Team Atlas make?",
    ]

    for question in questions:
        print(f"\n[질문] {question}")
        related_edges = retrieve_related_edges(question, graph)
        context = edges_to_context(related_edges)
        print(f"[검색된 관계]\n{context}")