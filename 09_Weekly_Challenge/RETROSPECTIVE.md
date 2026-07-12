## 트러블 슈팅 1 - LoRA와 QLoRA의 관계 정의 (순차 수정 vs 독립 분기)

### 문제 상황

- 로드맵 설계 초기에 LoRA로 학습한 결과물을 QLoRA로 "일부 수정"하는 작업으로 이해함
- 즉 Qwen Model → LoRA → QLoRA 순서로 이어지는 순차적 파이프라인으로 가정함
- 이 가정이 맞는지 검토가 필요했음

### 고려한 옵션

**고려 옵션**
| 관계 정의 | 내용 | 우리 과제 적합도 | 선택 여부 |
|---|---|---|---|
| 순차 수정 | LoRA adapter 결과물을 QLoRA용으로 일부 파라미터만 조정 | 낮음 (개념적으로 성립하지 않음) | 비선택 |
| **독립 분기** | 동일 원본 Qwen Model에서 LoRA, QLoRA를 각각 독립적으로 fine-tuning, 이후 비교하여 하나 선택 | **높음** | **선택** |

### 결정 및 이유

- 최종 결정: 독립 분기 구조
- 선택 이유:
  - LoRA와 QLoRA의 차이는 adapter 구조 차이가 아니라, adapter가 계산되는 대상인 기반 모델(base_model)의 정밀도 차이임 (LoRA는 bfloat16 기반 모델, QLoRA는 4-bit NF4 양자화 기반 모델)
  - 두 adapter는 서로 다른 forward pass 출력 분포(양자화 오차 포함 여부)를 보정하도록 학습되므로, 한쪽 결과물을 다른 쪽에 이어받는 방식은 성립하지 않음
  - 코드 레벨에서도 `from_pretrained` 호출 시 `torch_dtype` 또는 `quantization_config`만 다르고, 이후 LoRA config·학습 루프는 동일 — 즉 재사용 가능한 것은 adapter 결과물이 아니라 rank, alpha 등 하이퍼파라미터임

### 인사이트

- 두 기법의 이름이 유사하고 코드 구조 대부분이 겹치기 때문에 "순차적 확장"으로 오인하기 쉬우나, 실제로는 학습 대상(기반 모델의 정밀도)이 다른 별도 실행임을 확인
- 파이프라인을 다이어그램으로 그려볼 때, 화살표 하나가 실제로 "데이터/가중치 전달"을 의미하는지 "선택지 분기"를 의미하는지 구분하는 습관이 필요함

---

## 트러블 슈팅 2 - Before/After 성능 비교 기준점 설정

### 문제 상황

- 전체 파이프라인(Qwen → LoRA/QLoRA → PTQ → GGUF)의 각 단계마다 Before/After로 성능·메모리를 비교하기로 계획함
- 그러나 "Before"가 무엇을 가리키는지 기준이 명확하지 않았음
  - 매 단계 직전 단계의 결과물을 기준으로 할 것인가
  - 최초 원본 Qwen 모델을 고정 기준으로 할 것인가
- 기준을 하나만 선택하면 정보 손실이 발생함을 확인

### 고려한 옵션

**고려 옵션**
| 옵션 | 내용 | 장점 | 단점 | 평가 |
|---|---|---|---|---|
| A | 직전 단계 결과물을 Before로 사용 | 해당 단계에서 발생한 순수 변화량 파악 가능 | 전체 파이프라인 누적 효과 파악 불가 | 단독 사용 시 비추천 |
| B | 원본 Qwen을 고정 Before로 사용 | 전체 누적 효과 파악 가능 | 특정 단계에서 발생한 문제인지 이전 단계 누적인지 구분 불가 | 단독 사용 시 비추천 |
| **A+B 병행** | 원본 기준 표(최종 정리 표 1)로 진행하되, 최종 정리 단계에서 인접 행 차이로 직전 단계 기준 표(최종 정리 표 2)를 파생 계산 | 두 관점 모두 확보, 실시간 이중 측정 불필요 | 없음 | **선택** |

### 결정 및 이유

- 최종 결정: 최종 정리 표 1(원본 기준 누적 비교)만 매 단계마다 채우고, 최종 정리 표 2(직전 단계 기준 변화량)는 최종 정리 단계에서 최종 정리 표 1의 인접 행 차이로 파생 계산
- 선택 이유:
  - 실시간으로 두 종류의 표를 동시에 관리하면 측정 공수가 두 배로 들지만, 최종 정리 표 1만 정확히 채워두면 최종 정리 표 2는 산술 연산만으로 도출 가능해 효율적임
  - Phase 0에서 baseline을 반드시 기록해야 하는 이유가 이 결정과 직접 연결됨 — baseline 없이는 최종 정리 표 1 자체가 성립하지 않음

### 인사이트

- 비교 기준(reference point)을 명확히 정의하지 않고 측정부터 시작하면, 이후 데이터를 다시 수집해야 하는 상황이 발생할 수 있음
- 여러 관점의 비교가 필요할 때, 반드시 모든 관점을 실시간으로 측정할 필요는 없고 하나의 기준 데이터로부터 파생 가능한지 먼저 검토하는 것이 효율적임

---

## 트러블 슈팅 3 - ipynb 파일 구조 결정 (단일 파일 vs Phase별 분리)

### 문제 상황

- 챌린지 전체를 하나의 ipynb에서 진행할지, Phase별로 파일을 분리할지 결정이 필요했음
- 하나의 커널에서 LoRA용 base_model과 QLoRA용 base_model을 순차 실행할 경우, 이전 모델이 메모리에 남아있는 상태에서 다음 모델을 로드하게 되어 Mac M3 환경에서 메모리 부족 가능성이 우려됨

### 고려한 옵션

**고려 옵션**
| 옵션 | 내용 | 장점 | 단점 | 평가 |
|---|---|---|---|---|
| 파일 분리 | Phase별로 별도 ipynb 작성, json으로 metrics·checkpoint 전달 | 커널 재시작만으로 메모리 완전 초기화 | Phase 간 데이터 전달용 json 저장/로드 코드 추가 공수 발생 | 비추천 (공수 대비 이득 적음) |
| **단일 파일 + Empty Cache** | 하나의 ipynb 유지, Phase 전환 시점마다 `del`, `gc.collect()`, `empty_device_cache()` 실행 | 변수 재사용 편의성 유지, 추가 코드 최소화 | 완전한 메모리 초기화는 아님 (cache만 해제) | **선택** |

### 결정 및 이유

- 최종 결정: 단일 파일 + Phase 전환 시점마다 empty cache 셀 삽입
- 선택 이유:
  - json 저장/로드 구조를 새로 설계하는 공수가, 단순 empty cache 코드 세 줄 추가하는 공수보다 훨씬 큼
  - 최종 정리 표 1(원본 대비 누적 비교 metrics)을 하나의 변수(dict)로 계속 유지하며 append하는 방식이 단일 파일 구조에서 더 자연스럽게 이어짐
  - 메모리 부족 문제가 실제로 발생하면 그때 분리로 전환 가능 — 선(先) 단순 구조, 후(後) 필요 시 확장이 합리적 순서라고 판단

### 인사이트

- 관리 편의성(메모리 완전 초기화)과 구현 공수(전달 코드 추가) 사이의 트레이드오프를 판단할 때, 현재 시점에 실제로 발생한 문제가 아니라면 더 단순한 구조를 먼저 선택하고 필요 시 확장하는 접근이 효율적임
- `import json`처럼, 검토 과정에서 논의됐던 방식(분리 구조)의 흔적이 최종 결정(단일 파일)과 맞지 않는 코드로 남는 경우가 있으므로, 방향 전환 시 관련 코드를 함께 정리하는 습관이 필요함

---

## 트러블 슈팅 4 - 실행 환경 불일치로 인한 device 분기 및 Empty Cache 오류

- 이 트러블 슈팅은 단순한 코드 오류가 아니라, 로컬(Mac M3) 환경을 전제로 작성한 코드가 실제 실행 환경(Colab, CUDA)과 달라 발생한 문제를 기록한 내용

### 문제 상황

- 0-1 실행 결과 device가 `cpu`로 확인됨. 실제 실행 환경은 VSCode에서 Colab으로 원격 연결한 GPU(T4) 런타임이었음
- 0-5(Empty Cache) 실행 시 아래 에러 발생

```
RuntimeError: Cannot execute emptyCache() without MPS backend.
```

### 원인 분석

- 기존 device 분기 코드가 MPS(Mac 전용) 여부만 확인하고, CUDA 여부는 확인하지 않는 구조였음

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

- 이 코드는 이전 대화에서 로컬 Mac M3 환경을 전제로 작성된 것이며, 실행 환경이 Colab(CUDA)으로 바뀌었다는 정보가 이번에 처음 공유됨
- Empty Cache 코드 역시 동일한 이유로 MPS 전용 함수를 무조건 호출하도록 작성되어 있었음

```python
torch.mps.empty_cache()
```

- 즉 두 문제 모두 "코드 로직 자체의 결함"이 아니라, 전제한 실행 환경과 실제 실행 환경이 일치하지 않아서 발생한 문제였음

### 결정 및 대응

- device 분기를 CUDA → MPS → CPU 순서로 확인하도록 수정

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

- Empty Cache도 동일한 우선순위로 조건부 처리하는 함수로 분리

```python
def empty_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
```

- 수정 후 재실행 결과, device가 `cuda`로 정상 확인됨. latency 역시 23.39초 → 2.10초로 개선되어, 이전 수치가 GPU 미인식 상태에서 측정된 값이었음을 간접적으로 확인함

### 인사이트

- 코드를 작성할 때 전제한 실행 환경(로컬 Mac M3)과 실제 작업이 이루어지는 실행 환경(Colab, CUDA)이 다를 수 있다는 점을 놓쳤음
- 성능 지표(latency)가 비정상적으로 느리게 나올 때, 코드 로직 오류를 먼저 의심하기보다 **device가 실제로 의도한 하드웨어를 타고 있는지**부터 확인하는 것이 더 빠른 원인 파악 경로가 될 수 있음 — `print(device)`, `print(next(model.parameters()).device)` 같은 간단한 확인 코드를 초기 셀에 상시 배치하는 습관이 필요함
- 여러 실행 환경(로컬, Colab 등)을 오갈 가능성이 있는 프로젝트라면, 하드웨어 종속적인 코드(device 분기, cache 해제 등)는 처음부터 다중 환경을 고려한 조건부 구조로 작성하는 것이 재작업을 줄이는 방법임

---

## 트러블 슈팅 5 - 라이브러리 버전 불일치로 인한 반복적 ImportError

- 이 트러블 슈팅은 단일 오류가 아니라, 동일 원인(Colab 기본 환경의 구버전 라이브러리)이 서로 다른 두 라이브러리(torchao, bitsandbytes)에서 반복적으로 나타난 패턴을 기록한 내용

### 문제 상황

- 1-2(LoRA) 실행 중 `get_peft_model` 호출 시 아래 에러 발생

```
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
but only versions above 0.16.0 are supported
```

- torchao 업그레이드 후 1-2는 정상 동작했으나, 1-3(QLoRA) 실행 중 `BitsAndBytesConfig` 사용 시 유사한 에러 재발

```
ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes: pip install -U bitsandbytes>=0.46.1
```

- bitsandbytes를 pip으로 재설치했음에도 동일 에러가 한 번 더 발생

### 원인 분석

- 두 에러 모두 근본 원인은 동일함: Colab 기본 이미지에 사전 설치된 라이브러리 버전이, 현재 사용 중인 `peft`/`transformers` 버전이 요구하는 최소 버전보다 낮았음
  - torchao: 설치된 버전 0.10.0 < 요구 버전 0.16.0
  - bitsandbytes: 최초 미충족 상태 → 요구 버전 0.46.1 이상 필요
- bitsandbytes의 경우, pip install 로그상 `0.49.2`가 이미 만족된 상태(`Requirement already satisfied`)로 나왔음에도 동일 ImportError가 재발함
  - 원인은 **커널 재시작 누락**으로 판단됨. pip install은 파일 시스템에 새 버전을 설치하지만, 이미 실행 중인 Python 프로세스(Jupyter 커널)는 이전 시점에 import된 모듈 상태를 그대로 유지하고 있어, 재설치가 실제 실행 환경에 반영되지 않음
  - torchao 업그레이드 시에는 우연히(혹은 인지하지 못한 채) 재시작이 이루어져 정상 반영되었으나, bitsandbytes 때는 재시작 없이 재실행하여 동일 문제가 재현됨

### 결정 및 대응

- `!pip install -U torchao bitsandbytes --break-system-packages` 형태로 필요한 라이브러리를 한 번에 업그레이드
- 설치 후 반드시 **커널 재시작**을 거친 뒤 0-1부터 순서대로 재실행
- 향후 유사 문제를 앞단에서 예방하기 위해, 노트북 최상단(0-1 이전)에 환경 의존성 설치 셀("0-0")을 별도로 두는 방향을 고려함

### 인사이트

- pip install 로그에 `Successfully installed` 또는 `Requirement already satisfied`가 찍혀도, 그것이 곧바로 "현재 실행 중인 커널에 반영되었다"는 의미는 아님 — 설치와 반영은 별개의 단계이며, 반영을 위해서는 커널 재시작이 필요함
- 동일한 유형의 에러(라이브러리 버전 불일치)가 반복될 경우, 매번 개별적으로 대응하기보다 노트북 실행 초입에 의존성 설치 및 버전 고정을 한 번에 처리하는 구조가 반복 작업을 줄이는 방법이 됨
- Colab처럼 기본 이미지에 다양한 라이브러리가 사전 설치된 환경에서는, 최신 `transformers`/`peft` 기능을 쓰기 전에 관련 하위 의존성(quantization 관련 라이브러리)의 버전을 먼저 확인하는 습관이 필요함

---

## 트러블 슈팅 6 - Adapter Merge 실행 시점 재배치 (Phase 3 → Phase 2)

### 문제 상황

- 로드맵 설계 당시, Adapter Merge는 Phase 3(GGUF 변환 및 llama.cpp 추론)의 첫 단계(3-1)로 계획되어 있었음
- 그러나 Phase 2(PTQ)를 bitsandbytes 4-bit(NF4)로 진행하기로 확정하면서, 이 방식이 이미 메모리에 로드된 모델을 사후 변환하는 게 아니라 `from_pretrained` 호출 시점에 `quantization_config`를 지정해야만 적용된다는 제약이 드러남
- Phase 1의 LoRA 결과물은 base_model과 adapter가 분리된 `PeftModel` 상태라, 이 상태로는 `from_pretrained`로 다시 불러와 양자화할 단일 체크포인트가 존재하지 않는다는 문제가 확인됨

### 고려한 옵션

**고려 옵션**
| 옵션 | 내용 | 장점 | 단점 |
|---|---|---|---|
| 기존 계획 유지 | Adapter Merge를 Phase 3에서 그대로 진행 | 로드맵 변경 없음 | PTQ(Phase 2) 자체가 병합된 단일 체크포인트를 요구하므로 실행 불가능 |
| **재배치** | Adapter Merge를 Phase 2 초입(2-3-1)으로 이동, 병합된 모델을 그대로 PTQ 입력으로 사용 | PTQ 실행 가능, 이후 Phase 3은 이미 병합된 모델을 그대로 이어받음 | 로드맵 목차와 실제 실행 순서가 달라짐 |

### 결정 및 이유

- 최종 결정: Adapter Merge를 Phase 2(2-3-1)로 재배치
- 선택 이유:
  - PTQ 실행 자체가 병합된 단일 체크포인트를 선행 조건으로 요구하므로, 이 제약을 따르지 않으면 Phase 2 진행이 불가능함
  - Phase 3-1("Adapter Merge 여부 결정 및 실행")은 이미 Phase 2에서 병합이 완료된 상태이므로, 실질적으로 "병합된 모델임을 확인"하는 수준으로 축소됨

### 인사이트

- 로드맵 목차 작성 시점에는 각 Phase의 작업 단위를 개념적으로 나눴으나, 실제 구현에 들어가면 특정 기법(bitsandbytes 4-bit 등)의 API 제약이 Phase 간 순서를 강제하는 경우가 있음
- "이 작업은 다음 Phase에서 하기로 했다"는 계획이 있어도, 뒤 Phase가 실제로 무엇을 입력값으로 요구하는지 구현 직전에 재확인해야 하며, 필요하면 목차 자체를 유연하게 조정하는 것이 실행 가능한 순서를 유지하는 데 더 중요함

---

## 트러블 슈팅 7 - Phase 간 객체 재사용을 고려하지 않은 Empty Cache 설계

### 문제 상황

- 트러블 슈팅 6에 따라 Adapter Merge를 Phase 2(2-3-1)로 옮겨 진행하려는 시점에 아래 에러 발생

```
NameError: name 'lora_model' is not defined
```

- 원인은 Phase 1의 LoRA 학습 직후(1-2-5, Empty Cache)에서 실행한 아래 코드였음

```python
del lora_base_model, lora_model, lora_trainer
gc.collect()
empty_device_cache()
```

### 원인 분석

- 1-2-5 작성 당시에는 "Phase 1이 끝나면 LoRA 결과물을 더 이상 참조하지 않는다"는 전제로 `lora_model`까지 포함해 전부 삭제하도록 설계함
- 그러나 트러블 슈팅 6에서 확인된 대로, Adapter Merge가 Phase 2로 앞당겨지면서 `lora_model`을 Phase 2에서 다시 참조해야 하는 상황이 됨
- 즉 "해당 Phase가 끝났으니 여기서 만든 객체는 모두 지운다"는 단순 규칙으로 empty cache를 설계한 것이, 실제로는 이후 Phase가 이 객체를 입력으로 요구한다는 점을 반영하지 못한 것이 원인

### 결정 및 대응

- 1-2-5(LoRA Empty Cache)에서 `lora_model`은 삭제 대상에서 제외하고, 메모리 부담이 큰 `lora_base_model`, `lora_trainer`만 정리하도록 수정

```python
del lora_base_model, lora_trainer
gc.collect()
empty_device_cache()
```

- `qlora_model`은 이번 챌린지에서 다음 Phase로 선택되지 않았으므로 기존 삭제 코드를 유지하되, 추후 재참조가 필요해지면 동일한 방식으로 재검토하기로 함

### 인사이트

- 메모리 해제 코드는 "이번 Phase가 끝났다"는 시점 기준이 아니라, 전체 파이프라인에서 해당 객체가 이후 몇 단계까지 참조되는지를 먼저 확인하고 삭제 범위를 정해야 함
- 트러블 슈팅 6(Adapter Merge 재배치)과 이 문제는 같은 원인(로드맵상 계획과 실제 의존관계의 불일치)에서 파생된 연쇄적 결과이며, 상위 결정(작업 순서 변경)이 하위 구현(메모리 관리 코드)에도 영향을 미친다는 것을 보여주는 사례임

---

## 트러블 슈팅 8 - cmake 병렬 빌드 중 OOM(Out-Of-Memory) 발생

### 문제 상황

- llama.cpp를 CUDA 지원(`-DGGML_CUDA=ON`)으로 빌드하는 과정에서, 빌드가 43~45% 진행되던 중 다수의 `Killed` 메시지와 함께 컴파일 실패

```
gmake[3]: *** [.../conv2d-dw.cu.o] Error 137
Killed
```

### 원인 분석

- `cmake --build build --config Release -j`에서 `-j` 옵션에 값을 지정하지 않아, CPU 코어 수만큼 병렬 컴파일 작업이 동시에 실행됨
- CUDA 컴파일러(`nvcc`)는 `.cu` 파일 하나를 컴파일하는 데도 순간적으로 많은 메모리를 요구하는데, 이를 코어 수만큼 동시에 돌리면서 12GB RAM을 순식간에 소진
- Exit code 137(128+9, SIGKILL)은 Linux OOM Killer가 메모리 부족 상황에서 프로세스를 강제 종료했다는 신호였음

### 결정 및 대응

- `-j` 값을 명시적으로 낮춰 병렬 작업 수를 제한

```python
!cd llama.cpp && cmake --build build --config Release -j 2 --target llama-quantize llama-cli llama-perplexity
```

- 이미 컴파일된 오브젝트 파일은 캐시되어 재사용되므로, 처음부터 다시 빌드하지 않고 실패 지점부터 이어서 진행됨

### 인사이트

- `-j` 옵션에 값을 생략하면 "코어 수만큼 병렬 실행"이 기본값이 되는데, 이는 CPU 코어 수는 넉넉해도 RAM이 이를 못 받쳐주는 환경(특히 CUDA 컴파일처럼 단위당 메모리 요구가 큰 작업)에서는 오히려 독이 될 수 있음
- 크래시 발생 시점 이후에 `free -h`를 찍어봐도, 이미 새 프로세스가 시작된 뒤라 크래시 당시 상태를 보여주지 못함 — 실시간 원인 파악에는 한계가 있었고, 결국 빌드 로그 자체(`Killed`, `Error 137`)가 가장 직접적인 단서였음

---

## 트러블 슈팅 9 - IPython 매직 명령어와 셸 명령어 혼용 오류

### 문제 상황

- bash 계열 명령어(`git`, `pip`, `cmake` 등)를 노트북 셀에서 실행하며 `!`와 `%`를 혼용하다 아래 에러 발생

```
UsageError: Line magic function `%git` not found.
```

### 원인 분석

- `%`는 IPython이 자체 정의한 매직 명령어(`%cd`, `%pip`, `%time` 등) 전용 접두사이고, `git`처럼 일반 셸 프로그램은 매직 명령어로 존재하지 않음
- `!`는 셸 명령어 실행 전용, `%`는 IPython 매직 전용이라는 역할이 구분되어 있었는데, 이를 명확히 인지하지 못한 채 전환하며 사용함
- 다만 `pip`의 경우, `!pip`도 동작은 하지만 커널이 실제 사용 중인 Python 환경과 다른 환경에 설치될 수 있어 `%pip`가 권장되는 등, 명령어별로 권장 방식이 다름

### 결정 및 대응

- 순수 셸 명령어(`git`, `cmake`, `./build/bin/...`) → `!`
- IPython 커널 환경에 영향을 주는 `pip`, `cd`(세션 전체에 걸쳐 유지되어야 하는 경우) → `%`
- 멀티라인 백슬래시 연결이 필요한 경우 → `%%bash` 셀 매직으로 전환

### 인사이트

- `!`와 `%`는 비슷해 보이지만 실행 주체가 다름(하나는 셸 프로세스 위임, 하나는 IPython 자체 처리) — 이 차이를 알아두면 이후 유사한 명령어 실행 실패를 줄일 수 있음

---

## 트러블 슈팅 10 - TensorFlow/protobuf 임포트 충돌로 인한 0-2 재발 에러

### 문제 상황

- llama.cpp 빌드 관련 작업 이후, Phase 0(0-2)로 돌아와 Qwen 모델을 다시 로드하려는 시점에 아래 에러 발생

```
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

- `transformers`가 Qwen2 모델 클래스를 import하는 과정에서 부수적으로 `tensorflow`까지 로드를 시도하다 발생한 에러였음

### 원인 분석

- 이번 챌린지는 PyTorch만 사용하고 TensorFlow는 전혀 필요하지 않았으나, `transformers`가 내부적으로 `is_tf_available()`이 True로 판정되면 TF 관련 모듈까지 자동으로 import하는 경로를 탐
- `tensorflow`가 요구하는 `protobuf` 버전과 실제 설치된 `protobuf` 버전이 맞지 않아 충돌 발생
- 환경변수(`os.environ["USE_TF"] = "0"`)로 우회를 시도했으나, 이미 `transformers`가 이전 실행에서 import된 상태(`sys.modules`에 캐시됨)라 환경변수가 반영되지 않고 동일 에러가 재발함 — Reload Window는 VSCode 창만 새로고침할 뿐 Jupyter 커널 프로세스 자체를 재시작하지 않아, 커널 상태가 그대로 유지된 것이 원인이었음

### 결정 및 대응

- 환경변수 우회 대신, 이번 챌린지에 불필요한 `tensorflow` 자체를 제거

```python
%pip uninstall -y tensorflow
```

- 제거 후 반드시 **커널 재시작**(Restart Kernel, Reload Window와는 다른 동작) 후 0-1부터 재실행

### 인사이트

- 환경변수를 통한 라이브러리 동작 제어는 해당 라이브러리가 **처음 import되는 시점보다 먼저** 설정되어야 효과가 있음 — 이미 import된 세션에서는 아무리 환경변수를 바꿔도 반영되지 않음
- "Reload Window"와 "Restart Kernel"은 다른 동작이며, 커널 상태(이미 로드된 모듈)를 완전히 초기화하려면 반드시 Restart Kernel을 사용해야 함
- 근본적으로 불필요한 의존성(이번 경우 tensorflow)은 우회하기보다 제거하는 편이 재발 가능성을 낮춤

---

## 트러블 슈팅 11 - transformers 버전 문제로 인한 GGUF 변환 실패

### 문제 상황

- `convert_hf_to_gguf.py`로 merged_lora_model을 변환하는 도중, 토크나이저 로드 단계에서 아래 에러 발생

```
AttributeError: 'list' object has no attribute 'keys'
```

### 원인 분석

- `transformers`의 내부 메서드(`_set_model_specific_special_tokens`)가 `special_tokens`를 딕셔너리로 기대하고 `.keys()`를 호출했으나, 실제로는 리스트가 전달됨
- Colab에 설치된 `transformers` 버전이 지나치게 최신이라, llama.cpp의 변환 스크립트가 아직 검증하지 못한 최신 API 변경과 어긋난 것으로 추정됨 — 앞서 겪은 torchao, bitsandbytes 문제와 동일한 유형(Colab 기본 이미지의 최신 버전과 특정 도구 간 호환성 불일치)

### 결정 및 대응

- 변환 작업에 한해 `transformers`를 검증된 안정 버전으로 다운그레이드

```python
%pip install "transformers==4.46.3" --break-system-packages
```

- fine-tuning 및 병합 결과물(merged_lora_model)은 이미 디스크에 저장되어 있었으므로, Phase 0~2를 재실행할 필요 없이 변환 단계만 독립적으로 재시도 가능했음

### 인사이트

- Colab 기본 환경처럼 라이브러리 버전이 계속 최신화되는 환경에서는, 특정 도구(llama.cpp 변환 스크립트 등)가 아직 대응하지 못한 최신 API 변경으로 인한 충돌이 반복적으로 발생할 수 있음
- 학습 결과물을 디스크에 저장해두면(save_pretrained), 이후 단계에서 문제가 생겨도 전체 파이프라인을 처음부터 재실행하지 않고 문제 발생 지점부터만 복구할 수 있음 — 중간 산출물 저장의 중요성을 확인함

---

## 트러블 슈팅 12 - llama-cli 대화형 모드로 인한 셀 무한 대기

### 문제 상황

- `llama-cli`로 GGUF 모델 sanity check를 실행한 후, 응답이 정상적으로 출력되었음에도 셀이 5분 이상 계속 실행 중인 상태로 남음
- 정지 버튼을 눌러도 즉시 반응하지 않고 지연 후에야 종료됨

### 원인 분석

- `llama-cli`는 `-p`(prompt)로 초기 프롬프트를 주면 한 번 응답한 뒤, 기본적으로 대화형(interactive) 모드로 전환되어 다음 사용자 입력을 계속 대기함
- 노트북 셀 환경에서는 이 대화형 입력을 칠 방법이 없어, 응답이 이미 정상 완료되었음에도 셀이 끝나지 않고 대기 상태로 남은 것
- Jupyter의 Interrupt(정지 버튼)가 곧바로 반응하지 않았던 것은, `llama-cli`가 C++로 컴파일된 외부 프로세스이고 stdin을 블로킹 방식으로 읽고 있어 신호 전달과 처리에 지연이 있었기 때문으로 추정됨

### 결정 및 대응

- `-no-cnv`(비대화형) 옵션을 추가해 한 번 응답 후 자동 종료되도록 수정
- 이후 metrics 기록 및 반복 실행은 Python 바인딩(`llama_cpp.Llama`)으로만 진행 — CLI는 최초 1회 sanity check 용도로만 남기고, 자동화가 필요한 반복 작업에서는 배제함

### 인사이트

- CLI 도구는 원래 사람과의 대화형 상호작용을 위해 설계된 경우가 많아, 노트북처럼 자동화된 파이프라인에 그대로 사용하면 이번과 같은 대기 문제가 발생하기 쉬움 — 자동화 맥락에서는 처음부터 배치/비대화형 실행을 지원하는 옵션이나 별도 바인딩을 우선 고려하는 것이 안전함
- 같은 llama.cpp 계열 실행 파일이라도 목적(대화형 CLI vs 배치성 계산 도구)이 다르면 동작 방식이 완전히 다를 수 있어, 이전 문제(대화형 모드 행업)의 해결책(예: timeout)을 다른 실행 파일에 검증 없이 그대로 적용하면 불필요한 코드가 남을 수 있음(트러블 슈팅 13 참고)

---

## 트러블 슈팅 13 - llama-perplexity 최소 토큰 요구량 미달

### 문제 상황

- `llama-perplexity`로 evaluation_text_list(20개 문장)의 perplexity를 측정하려는 시점에 아래 에러 발생

```
E perplexity: you need at least 1024 tokens to evaluate perplexity with a context of 512
E perplexity: the data file you provided tokenizes to only 383 tokens
```

### 원인 분석

- `llama-perplexity`는 텍스트를 슬라이딩 윈도우 방식으로 여러 청크로 나누어 평가하기 때문에, 최소한 context 크기(기본 512)의 2배 이상 토큰 수를 요구함
- evaluation_text_list를 파일로 저장한 결과가 383토큰뿐이라 최소 요구치(1024)에 미달
- 지금까지 사용해온 `measure_perplexity`(Hugging Face 기반, 문장별 개별 평가 후 평균)와 `llama-perplexity`(전체 텍스트를 연속 시퀀스로 슬라이딩 윈도우 평가)는 계산 방식 자체가 근본적으로 다르다는 점이 이 과정에서 드러남

### 결정 및 대응

- evaluation_text_list를 인위적으로 늘리는 대신, context 크기(`-c`)를 128로 낮춰 최소 요구 토큰 수를 데이터 크기에 맞춤

```python
!cd llama.cpp && ./build/bin/llama-perplexity -m ../qwen_daysync_q4_k_m.gguf -f ../evaluation_text.txt -c 128
```

- 이 방식으로 측정된 GGUF의 perplexity(5.5704)는 다른 Phase의 perplexity(baseline 13.66, LoRA 13.10, PTQ 22.44)와 **계산 방식이 달라 직접 비교 불가**하다는 점을 최종 정리 표 1에 각주로 명시

### 인사이트

- 동일한 이름(perplexity)의 지표라도, 측정 도구가 다르면 계산 방식(개별 문장 평균 vs 슬라이딩 윈도우 연속 평가)이 달라 절대값을 직접 비교할 수 없는 경우가 있음 — 지표 이름만 보고 동일 선상에서 비교하면 잘못된 결론(예: "GGUF가 압도적으로 우수하다")에 이를 수 있음
- 도구가 요구하는 최소 조건(이번 경우 최소 토큰 수)을 데이터 크기에 맞추는 것과, 데이터를 도구의 요구치에 맞춰 늘리는 것 중 무엇이 더 타당한 선택인지는 상황에 따라 다르며, 이번에는 평가 데이터의 원본 의미를 훼손하지 않는 전자(context 축소)를 선택함

---

## 트러블 슈팅 14 - NF4 양자화 구성에서 Double Quantization 누락 (QLoRA, PTQ 공통)

### 문제 상황

- Phase 1(1-3-1)에서 QLoRA를 구현할 때 사용한 `BitsAndBytesConfig`에 `bnb_4bit_use_double_quant`가 지정되지 않음

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

- Phase 2(2-3-2, PTQ)에서도 별도로 작성된 `BitsAndBytesConfig`에 동일하게 `bnb_4bit_use_double_quant`가 지정되지 않아, PTQ 단계 역시 Double Quantization 없이 진행됨 (Phase 1의 설정을 재사용한 것이 아니라, 서로 무관하게 작성된 코드에서 같은 누락이 독립적으로 발생함)
- 결과적으로 최종 정리 표 1에서 PTQ의 memory 절감폭이 이론적 최대치(1/4)에 못 미쳤던 원인(quantization constant 저장 오버헤드) 중 일부가 이 누락에서 비롯되었을 가능성이 확인됨

### 원인 분석

- Double Quantization은 NF4 등 4-bit 양자화를 적용할 때 quantization constant(scale 값)까지 추가로 압축해 메모리를 더 아끼는 부가 옵션(`bnb_4bit_use_double_quant`)이며, 이 옵션의 기본값이 `False`라 명시하지 않으면 조용히 꺼진 채로 실행됨
- 이 챌린지에서 NF4 4-bit 양자화가 적용된 지점은 두 곳(Phase 1의 QLoRA, Phase 2의 PTQ)이었고, 두 곳 모두 이 옵션이 누락됨
  - Phase 1(1-3-1, QLoRA): 기반 모델을 4-bit로 로드하며 학습을 진행하는 과정에서 누락 — QLoRA 논문이 권장하는 표준 구성(NF4 + Double Quantization + bfloat16 compute dtype)을 검토하지 못함. 다만 QLoRA는 Phase 1에서 최종 선택되지 않았으므로(LoRA가 선택됨), 이 누락이 이후 단계에 직접 영향을 준 것은 아님
  - Phase 2(2-3-2, PTQ): 병합된 LoRA 결과물(merged_lora_model)을 4-bit로 재로드하며 별도로 작성한 `BitsAndBytesConfig`에서 동일하게 누락 — Phase 1의 설정을 재사용한 것이 아니라, QLoRA와 무관하게 새로 작성된 코드에서 같은 옵션 검토가 독립적으로 빠진 것
- 즉 최종 결과(최종 정리 표 1)의 memory 수치에 실제 영향을 준 것은 Phase 2(PTQ) 쪽 누락이며, Phase 1(QLoRA) 쪽 누락은 어차피 선택되지 않은 경로라 결과에 직접 반영되지 않음

### 결정 및 대응

- Double Quantization 반영 여부를 결정 검증 항목으로 별도 등재하고, 추후 과제로 남김

```python
ptq_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # 추후 반영 시 추가할 옵션
)
```

- 이번 챌린지에서는 즉시 재실행하지 않고, memory 절감폭을 재비교할 향후 과제로 분리함

### 인사이트

- 기본값이 `False`인 옵션은 명시적으로 켜지 않아도 에러 없이 조용히 넘어가기 때문에, "표준 구성을 따랐는지"가 실행 결과만으로는 드러나지 않음
- 서로 다른 Phase에서 관련 없이 독립적으로 작성된 코드임에도 동일한 누락이 반복되었다는 것은, 코드 재사용 여부와 무관하게 애초에 "이 양자화 방식에서 검토해야 할 표준 옵션 목록" 자체가 사전에 정리되어 있지 않았다는 게 더 근본적인 원인임을 시사함

---

## 트러블 슈팅 15 - Double Quantization과 QLoRA 정의의 혼동

### 문제 상황

- 트러블 슈팅 14를 확인하는 과정에서, "Phase 2(PTQ)에 Double Quantization을 적용하면 LoRA가 QLoRA로 바뀌는 것 아닌가"라는 의문이 제기됨
- 즉 Double Quantization의 적용 여부가 QLoRA를 QLoRA이게 만드는 정의 조건 중 하나인지, 아니면 QLoRA와 무관하게 켤 수 있는 부가 옵션인지 불명확했음

### 원인 분석

- QLoRA의 정의(자료 기준: "4-bit NF4로 양자화된 기반 모델은 고정하고, 16-bit LoRA 어댑터만 학습해 메모리 사용량을 줄이는 PEFT 기법")를 분해하면 핵심 조건은 다음과 같음
  1. 기반 모델이 **학습이 시작되기 전에 이미** 4-bit NF4로 양자화되어 있어야 함
  2. 그 양자화된 기반 모델은 고정(freeze)
  3. LoRA adapter만 16-bit로 학습
- 이 세 조건 중 어디에도 Double Quantization은 필수 조건으로 명시되어 있지 않음 — Double Quantization은 "4-bit 양자화를 적용할 때 quantization constant까지 추가로 압축해 메모리를 더 아끼는" 부가적인 세부 구현 옵션일 뿐, QLoRA만의 전유물이 아니며 순수 PTQ에도 독립적으로 적용 가능함
- 반면 LoRA와 QLoRA를 실제로 가르는 것은 "**언제** 양자화가 개입하는가"임
  - LoRA: bfloat16 기반 모델 위에서 학습 (양자화 없음)
  - QLoRA: 이미 4-bit로 양자화된 기반 모델 위에서 학습 (학습 시점에 양자화가 전제됨)
- Phase 2의 PTQ는 이미 fine-tuning이 끝난 LoRA 결과물을 **사후에** 4-bit로 변환하는 것이므로, 여기에 Double Quantization을 추가해도 "학습 시점에 양자화된 상태"라는 QLoRA의 조건 자체를 충족시키지 못함 — 즉 Double Quantization을 켜더라도 이는 여전히 PTQ이지 QLoRA로 전환되는 것이 아님

### 결정 및 대응

- Double Quantization은 QLoRA 여부와 독립적인 옵션으로 취급하기로 함 — Phase 2(PTQ)에 이 옵션을 추가로 켜더라도, 이는 PTQ의 메모리 효율을 개선하는 조치일 뿐 Fine-Tuning 단계의 선택(LoRA vs QLoRA)을 소급해서 바꾸는 것이 아님을 확인
- 트러블 슈팅 14의 결정 검증 항목은 "PTQ의 memory 효율 개선 여부"로만 좁혀서 유지하고, "LoRA를 QLoRA로 바꾸는 것"이라는 오해는 배제함

### 인사이트

- 두 기법(QLoRA)의 이름과, 그 기법에 흔히 함께 등장하는 부가 최적화 옵션(Double Quantization)을 혼동하기 쉬움 — "A 기법에서 자주 쓰이는 옵션 B"가 "A 기법의 정의 조건"과 반드시 같지는 않음
- 기법을 정의하는 핵심 조건(이번 경우 "학습 시점에 양자화가 이미 적용되어 있는가")과, 그 기법을 더 효율적으로 만드는 부수적 구현 디테일을 구분해서 이해하는 것이, 유사한 개념들이 얽혀 있는 최적화 분야에서 특히 중요함
- "이 옵션을 켜면 다른 기법이 되는가"라는 질문이 떠오를 때는, 그 기법의 정의를 조건 단위로 다시 분해해서, 해당 옵션이 그 조건들 중 하나에 해당하는지 직접 대조해보는 것이 혼동을 해소하는 가장 확실한 방법임
