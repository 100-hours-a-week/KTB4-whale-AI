# Virtual document design specification

## (txt/md) Nimbusflow Manual([살펴보기](./nimbusflow_manual.md))

| 속성      | 값                                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------------- |
| 문서 유형 | 가상 소프트웨어 제품 매뉴얼                                                                                 |
| 형식      | Markdown (.md)                                                                                              |
| 구성      | 제품 개요, 설치/설정 변경, API 사용법, 에러 코드, FAQ                                                       |
| 목적      | Gemma 4가 절대 모를 가상의 고유 정보(제품명, API 파라미터명, 에러코드 등)를 포함시켜 RAG 효과를 명확히 검증 |

(가상 문서 설계 핵심) 이 메뉴얼에는 Gemma 4가 절대 알 수 없는 고유 정보들이 의도적으로 박혀있어, RAG가 제대로 동작하는지 여부를 비교적 명확하게 확인할 수 있음.

**RAG 검증용 핵심 정보(Verification Anchors)**

| 정보                      | 값                               | 검증 시 사용                                              |
| ------------------------- | -------------------------------- | --------------------------------------------------------- |
| 제품 내부 코드네임        | "Project Driftwood"              | RAG 없이 물으면 "모른다"고 답해야 정상                    |
| `engine_mode` 기본값      | `solo`                           | 문서 없이는 추측 불가능한 고유 설정값                     |
| `hybrid_sync` 전환 임계값 | Drift Score 0.73                 | 매우 구체적인 수치 — 환각(hallucination) 여부 판별에 좋음 |
| 기본 API 포트             | 8842                             | 일반적인 포트(80, 443, 8080)가 아니므로 추측 불가         |
| 에러 코드 NF-227 의미     | "Drift Score 계산 타임아웃(4초)" | 코드와 의미를 정확히 매칭하는지 검증                      |
| 유지보수 회사명           | "Driftwood Systems, 2022년 설립" | 완전히 가상의 정보                                        |

이 문서를 통해 다음과 같은 질문으로 RAG 효과를 검증할 수 있음.

- "What is NimbusFlow's internal codename?" (RAG는 모르는 정보)
- "What does error code NF-227 mean?" (RAG 없이는 지어낼 가능성)
- "What is the default value of checkpoint_interval_sec?" (구체적 수치 - 환각 여부 명확히 드러남)
