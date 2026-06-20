# RAG Architecture Project

**Document Loading 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | 파일 경로(file path) — 문자열 |
| Output (출력) | Raw text — 단일 문자열(string) |
| 책임(Responsibility) | 파일을 읽어서 텍스트로 변환하는 것까지만. Chunking(분할)은 다음 단계의 책임이므로 여기서 하지 않음 |

**Fixed-size Chunking 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | Raw text — 단일 문자열 |
| Output (출력) | Chunk 리스트 — `list[str]` |
| 핵심 파라미터 | `chunk_size` (chunk당 문자 수), `chunk_overlap` (인접 chunk 간 겹치는 문자 수) |
| 책임(Responsibility) | 텍스트를 정해진 길이로 자르는 것까지만. Embedding은 다음 단계의 책임 |
