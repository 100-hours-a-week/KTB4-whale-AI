# RAG Architecture Project

**Document Loading 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | 파일 경로(file path) — 문자열 |
| Output (출력) | Raw text — 단일 문자열(string) |
| 책임(Responsibility) | 파일을 읽어서 텍스트로 변환하는 것까지만. Chunking(분할)은 다음 단계의 책임이므로 여기서 하지 않음 |
