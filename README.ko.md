# Awesome RTX 5090

[English](README.md) | **한국어**

> NVIDIA GeForce RTX 5090에서 검증된 재현 가능한 벤치마크, 모델 실행법, 추론 엔진, 생성형 AI 워크플로와 시스템 도구를 정리한 목록입니다.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub topic](https://img.shields.io/badge/GitHub%20topic-rtx--5090-181717?logo=github)](https://github.com/topics/rtx-5090)
[![License: CC0](https://img.shields.io/badge/license-CC0--1.0-blue.svg)](LICENSE)

RTX 5090 전용 측정 결과, 설정 방법 또는 Blackwell `sm_120` 지원 자료가 있는 항목을 선별했습니다. 카드 이름을 단순히 언급한 프로젝트는 포함하지 않습니다.

> [!NOTE]
> 모델, 양자화, 컨텍스트, 입출력 길이, 백엔드와 샘플링 설정이 같지 않다면 아래 벤치마크 수치를 직접 비교할 수 없습니다. 정확한 측정 조건은 각 출처에서 확인하세요.

## 목차

- [빠른 참고](#빠른-참고)
- [모델 매트릭스](#모델-매트릭스)
- [벤치마크와 평가](#벤치마크와-평가)
- [LLM 추론과 서빙](#llm-추론과-서빙)
- [코딩 에이전트와 개발 도구](#코딩-에이전트와-개발-도구)
- [Windows와 WSL](#windows와-wsl)
- [이미지, 영상, 3D와 오디오](#이미지-영상-3d와-오디오)
- [CUDA와 ML 라이브러리](#cuda와-ml-라이브러리)
- [하드웨어, 전력과 가상화](#하드웨어-전력과-가상화)
- [토픽 밖의 프로젝트 찾기](#토픽-밖의-프로젝트-찾기)
- [선정 기준](#선정-기준)
- [기여](#기여)

## 빠른 참고

| 항목 | 데스크톱 RTX 5090 |
|---|---|
| 아키텍처 | NVIDIA Blackwell, GB202 |
| 컴퓨트 능력 | `sm_120` / 기능에 따라 `sm_120a` |
| VRAM | 32 GB GDDR7 |
| 일반적인 로컬 AI 환경 | 네이티브 Windows, WSL2 또는 Linux |
| 일반적인 추론 형식 | GGUF, FP8, NVFP4, BF16 |
| 먼저 볼 자료 | [종합 벤치마크](https://github.com/patrickwhelan-uk/llm-inference-benchmarks), [모델 품질 평가](https://github.com/gisenberg/local-model-eval), [Windows 가이드](https://github.com/OnlyTerp/windows-is-fine-for-llms) |

노트북용 RTX 5090은 메모리 용량과 전력 한도가 다릅니다. 데스크톱 측정 결과가 그대로 적용된다고 가정하지 않으며 별도 결과로 취급합니다.

## 모델 매트릭스

이 표는 공개된 측정 결과를 찾아가기 위한 색인이며 순위표가 아닙니다. “보고된 결과”는 링크된 프로젝트가 공개한 수치를 그대로 옮긴 것입니다.

| 모델 | 양자화 / 정밀도 | 백엔드 | 보고된 RTX 5090 결과 | 컨텍스트 / 작업 | 출처 |
|---|---|---|---:|---|---|
| Ministral 3 3B | GGUF(출처 설정) | LM Studio | 265.2 tok/s, 최대 VRAM 7.4 GB | 프롬프트 512토큰, 생성 128토큰 | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Llama 3.1 8B | FP16 | vLLM | 동시성 1에서 82 tok/s, 동시성 4에서 합계 304 tok/s | 짧은 프롬프트, 최대 출력 256토큰, smoke test | [llm-gpu-benchmark](https://github.com/DaytonLecoq/llm-gpu-benchmark#current-results-fp16-smoke-test) |
| Llama 3.1 8B | GGUF(출처 설정) | LM Studio | 140.4 tok/s, 최대 VRAM 12.1 GB | 프롬프트 512토큰, 생성 128토큰 | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Qwen 2.5 7B | GGUF(출처 설정) | LM Studio | 145.6 tok/s, 최대 VRAM 11.4 GB | 프롬프트 512토큰, 생성 128토큰 | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Qwen 2.5 Coder 32B | GGUF(출처 설정) | LM Studio | 72.2 tok/s, 최대 VRAM 23.2 GB | 프롬프트 512토큰, 생성 128토큰 | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Qwen 3 8B | Q8_0 | imp | 385.4 tok/s | 같은 GGUF와 플래그를 사용한 단일 스트림 디코드 비교 | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Qwen 3 14B | Q6_K | imp | 162.5 tok/s | 같은 GGUF와 플래그를 사용한 단일 스트림 디코드 비교 | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Qwen 3 30B-A3B | Q4_K_M | imp | 305.5 tok/s | 같은 GGUF와 플래그를 사용한 단일 스트림 디코드 비교 | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Qwen 3.5 35B-A3B | Q4_K_M | Ollama | 최고 145.6 tok/s, 131K에서 120 tok/s | Windows 11, 최대 196K 시험 | [5090-windows11-benchmarks](https://github.com/taco-devs/5090-windows11-benchmarks/tree/master/qwen3.5-35b-a3b) |
| Qwen 3.6 35B-A3B | UD-Q4_K_M | imp | 287.9 tok/s | 같은 GGUF와 플래그를 사용한 단일 스트림 디코드 비교 | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Qwen 3.8 27B | NVFP4 | vLLM / NInfer | 프로젝트 주장 약 139 tok/s | MTP를 적용한 262K 구성 | [Qwen5090](https://github.com/Ark0N/Qwen5090) |
| Qwen 3.8 27B | NVFP4 가중치 + KV | vLLM | 즉시 실행 가능한 서빙 구성, 대표 처리량 수치 없음 | 262K, MTP-3, 최대 8개 스트림 | [vllm-sm120-nvfp4-mtp](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp) |
| Gemma 4 26B-A4B | Q6_K | llama.cpp / TurboQuant 포크 | 161.9 tok/s, 코딩 테스트 16/17 | 품질과 처리량 평가 | [local-model-eval](https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_5090.md) |
| Gemma 4 31B | Q4_K_M | llama.cpp / TurboQuant 포크 | 50.3 tok/s, 코딩 테스트 17/17 | 28 GB에서 전체 262K 보고 | [local-model-eval](https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_5090.md) |
| GPT-OSS 20B | MXFP4 | imp | 382.7 tok/s | 같은 모델과 플래그를 사용한 단일 스트림 디코드 비교 | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Nemotron 3 Nano | GGUF(출처 설정) | LM Studio | 237.5 tok/s, 최대 VRAM 26.9 GB | 프롬프트 512토큰, 생성 128토큰 | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |

모든 수치는 빠르게 바뀌는 소프트웨어의 특정 시점 결과입니다. “프로젝트 주장”으로 표시된 행은 유용한 설정 자료이지만 이 목록이 조건을 통일하거나 독립적으로 검증한 결과가 아닙니다.

## 벤치마크와 평가

- [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks) — 여러 모델의 디코드, 프롬프트 평가, TTFT, VRAM과 전력을 측정하는 재현 가능한 LM Studio/llama.cpp 도구입니다.
- [local-model-eval](https://github.com/gisenberg/local-model-eval) — 코딩 품질과 처리량, 컨텍스트 용량, 양자화 영향을 함께 평가하며 원시 실험 구조와 5090 순위를 제공합니다.
- [5090-windows11-benchmarks](https://github.com/taco-devs/5090-windows11-benchmarks) — Windows 11에서 컨텍스트 확장, needle 검색, VRAM 한계와 Ollama/vLLM을 비교합니다.
- [llm-gpu-benchmark](https://github.com/DaytonLecoq/llm-gpu-benchmark) — 지연시간 백분위, 사용률, VRAM, 온도와 시스템 지표를 수집하는 vLLM 동시성 측정 도구입니다. 현재 공개된 결과는 소규모 FP16 smoke test입니다.
- [NV-benchmark](https://github.com/QuanTuring-AI/NV-benchmark) — NIM/TensorRT-LLM과 Ollama의 처리량, TTFT, 지연시간 및 guardrail 비용을 비교합니다.
- [ecocompute-dynamic-eval](https://github.com/hongping-zh/ecocompute-dynamic-eval) — RTX 5090 양자화 실험을 포함해 모델 정확도, 비용과 에너지·탄소 지표를 비교합니다.
- [gemma-rig](https://github.com/Stoneforge-Labs/gemma-rig) — RTX 5090 한 장에서 측정한 Gemma 서빙, 평가와 에이전트 작업 결과입니다.
- [benchmark-rtx5090](https://github.com/pendakwahteknologi/benchmark-rtx5090) — Qwen 2.5 네 가지 크기와 세 가지 양자화를 대상으로 프롬프트·디코드 속도, 전력 효율과 비용을 측정합니다. `rtx-5090` 토픽 밖에서 발견했습니다.

## LLM 추론과 서빙

- [imp](https://github.com/kekzl/imp) — `sm_120a`에 맞춘 C++/CUDA 추론 엔진으로 llama.cpp 및 vLLM과의 단일 스트림·동시 처리 벤치마크를 제공합니다.
- [sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer) — MoE 모델과 소비자·엣지 Blackwell GPU에 초점을 맞춘 CUDA 추론 런타임이며 모델별 RTX 5090 결과 파일을 포함합니다.
- [blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm) — vLLM, FlashInfer와 CUTLASS용 SM120 패치 및 NVFP4 커널입니다.
- [vllm-nvfp4-docker](https://github.com/ckienstra/vllm-nvfp4-docker) — Blackwell의 NVFP4 KV 캐시를 지원하는 vLLM Docker 빌드·서빙 구성입니다.
- [vllm-sm120-nvfp4-mtp](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp) — Qwen 3.8 27B, NVFP4 KV, MTP-3, 도구 호출과 긴 컨텍스트를 위한 버전 고정 커뮤니티 vLLM 이미지입니다.
- [club-3090](https://github.com/noonghunna/club-3090) — 3090/4090/5090 한 장 또는 두 장에서 Qwen과 Gemma를 서빙하는 커뮤니티 구성입니다.

## 코딩 에이전트와 개발 도구

- [qwen3.8-27b-claude-code-desktop-bridge](https://github.com/Maharajahu/qwen3.8-27b-claude-code-desktop-bridge) — RTX 5090 한 장에서 로컬 Qwen 3.8 27B Q6 GGUF를 Claude Code Desktop 또는 CLI에 연결합니다. Anthropic 요청을 OpenAI 형식으로 변환하는 bridge가 reasoning, vision, streaming과 tool call을 보존하며, 텍스트 200K와 vision 128K 프로필을 설명합니다.
- [Qwen5090](https://github.com/Ark0N/Qwen5090) — Qwen 3.8 27B, OpenAI 호환 API와 Claude Code·DeepSeek Harness 같은 로컬 코딩 에이전트를 위한 Windows 원클릭 및 Linux 스크립트 설정입니다.

## Windows와 WSL

- [windows-is-fine-for-llms](https://github.com/OnlyTerp/windows-is-fine-for-llms) — 네이티브 Windows와 WSL 선택, 디스플레이 GPU의 TDR 안정성, 메모리 한계와 llama.cpp 서빙을 설명합니다.
- [comfyui-triton-and-sageattention-installer](https://github.com/DazzleML/comfyui-triton-and-sageattention-installer) — Windows와 RTX 50 시리즈를 지원하는 크로스플랫폼 Triton·SageAttention 설치 도구입니다.
- [triton-blackwell-windows-pipeline](https://github.com/Arcen-c1c/triton-blackwell-windows-pipeline) — FP8 검증과 벤치마크 결과를 포함한 Windows용 `sm_120a` Triton 빌드 파이프라인입니다.

## 이미지, 영상, 3D와 오디오

- [ComfyUI-Pixal3D](https://github.com/dreamrec/ComfyUI-Pixal3D) — 단일 이미지에서 텍스처가 있는 PBR 메시를 생성하는 ComfyUI 통합으로 RTX 30/40/50 시리즈에서 시험했습니다.
- [local-ai-video-generator](https://github.com/carrotProgrammer/local-ai-video-generator) — RTX 5090에서 검증한 Wan 2.2와 MiniMax H3 기반 로컬 ComfyUI 영상 워크플로입니다.
- [csm-rtx5090](https://github.com/D3velop-llc/csm-rtx5090) — CUDA 그래프와 `torch.compile`로 Blackwell에 최적화한 CSM-1B 스트리밍 TTS 파이프라인입니다.
- [fish-s2-rtx](https://github.com/Genesis1231/fish-s2-rtx) — vLLM-Omni 기반 OpenAI 호환 OpenAudio S2-Pro 스트리밍 TTS 및 음성 복제입니다.
- [mpv-god-preset](https://github.com/GoldenSample/mpv-god-preset) — TensorRT 업스케일링, RIFE 보간, HDR 및 4K/8K 재생을 측정한 mpv/VapourSynth 설정입니다.
- [ComfyUI-Qlip](https://github.com/TheStageAI/ComfyUI-Qlip) — FLUX.2 Klein, Z-Image-Turbo, Wan 2.2 등 이미지·영상 모델의 RTX 5090 측정 결과가 있는 컴파일형 확산 엔진입니다. 토픽 밖에서 발견했습니다.
- [rtx509032g-minimax-h3-comfyui](https://github.com/Nuos/rtx509032g-minimax-h3-comfyui) — 전체 실행시간 결과, 기계 판독 데이터와 시스템 메모리 분석을 제공하는 재현 가능한 MiniMax H3 텍스트·이미지 영상화 워크플로입니다.
- [comfyui-vae-float32](https://github.com/AndreiOrehov/comfyui-vae-float32) — VRAM spill 동작을 포함해 RTX 5090에서 측정한 LTX 영상 VAE 정밀도와 타일링 실험입니다.

## CUDA와 ML 라이브러리

- [FastFlashAttention](https://github.com/AlcAI-Haven/FastFlashAttention) — Blackwell `sm_120`에 맞춘 결정적 backward 지원 exact BF16 flash attention입니다.
- [MinkowskiEngine](https://github.com/alpsaur/MinkowskiEngine) — CUDA 12.8+, Blackwell, PyTorch 2.x와 최신 NumPy용으로 갱신하고 미리 빌드한 wheel을 제공하는 커뮤니티 포크입니다.
- [NATTEN Windows SM120 wheels](https://github.com/mmluankoko/NATTEN-0.21.6-Windows-Python3.13-Torch2.10-CUDA13.0-sm120) — Python 3.13, PyTorch 2.10, CUDA 13.0과 `sm_120`용 Windows NATTEN wheel입니다.
- [pytorch-tensorflow-gpu](https://github.com/dconsorte/pytorch-tensorflow-gpu) — PyTorch와 TensorFlow, CUDA 12.8, Blackwell 지원을 합친 Docker 환경입니다. 포크이므로 적용 전 upstream과의 차이를 확인하세요.

## 하드웨어, 전력과 가상화

- [astral-hwmon](https://github.com/ksokolowski/astral-hwmon) — ASUS ROG Astral 카드의 12VHPWR 핀별 상태를 Linux `hwmon` 센서로 노출하고 보호 도구를 제공합니다.
- [ConnectorWatch](https://github.com/Shaderx/ConnectorWatch) — 대시보드, 트레이 알림과 휴대형 데몬을 갖춘 실험적인 Windows 입력 전압 추세 모니터입니다.
- [gpu-tuner](https://github.com/xdzleo/gpu-tuner) — NVIDIA RTX 40/50 시리즈의 언더볼트, 오버클럭과 효율 지점을 탐색합니다.
- [vgpu-unlock-blackwell](https://github.com/bird/vgpu-unlock-blackwell) — 소비자용 Blackwell vGPU 활성화 연구입니다. README에 현재 펌웨어·하드웨어 차단 요소가 기록되어 있으며 작동하는 unlock 도구는 아닙니다.
- [macuda](https://github.com/Davinchy/macuda) — Apple Silicon macOS에서 RTX 5090을 Thunderbolt eGPU로 실행하는 실험적 사용자 공간 NVIDIA 드라이버와 CUDA/cuBLAS shim입니다. 정확성 검사와 네이티브 비교 결과가 있습니다.

## 토픽 밖의 프로젝트 찾기

GitHub 토픽은 좋은 시작점이지만, 뛰어난 결과 중 상당수는 토픽을 달지 않습니다. 2026-09-18 조사에서 `rtx-5090` 토픽을 제외하고도 `RTX 5090`을 언급한 저장소가 GitHub 저장소 검색에 9천 개 이상, README 코드 검색에 6천 개 이상 나타났습니다. 대부분은 단순 환경 표기이므로 발견 단계와 증거 검증 단계를 분리해야 합니다.

### GitHub 검색

- [“RTX 5090” 저장소 README 검색](https://github.com/search?q=%22RTX+5090%22+in%3Areadme&type=repositories) — 최근 업데이트 순으로 정렬한 다음 README를 확인하는 넓은 검색입니다.
- [README 코드 검색](https://github.com/search?q=%22RTX+5090%22+path%3AREADME.md&type=code) — 토픽이나 설명에 표시되지 않은 하드웨어 표와 벤치마크 기록을 찾습니다.
- [결과 파일 검색](https://github.com/search?q=%22RTX+5090%22+%28path%3Aresults+OR+path%3Abenchmarks%29&type=code) — 소개 문구 대신 커밋된 측정 자료를 찾습니다.
- [`sm_120` README 검색](https://github.com/search?q=sm_120+path%3AREADME.md&type=code) — 5090을 직접 쓰지 않고 “RTX 50 시리즈”라고 표현한 Blackwell 호환성 작업을 찾습니다.
- 이슈와 Discussions에서 `no kernel image`, `compute_120a`, `unsupported gpu architecture` 같은 정확한 오류 문구를 검색하면 README보다 먼저 공개된 해결책을 찾을 수 있습니다.

유용한 GitHub CLI 검색식:

```bash
# 카드를 언급하지만 토픽은 사용하지 않는 저장소
gh api -X GET search/repositories \
  -f q='"RTX 5090" in:name,description,readme -topic:rtx-5090' \
  -f sort=updated -f order=desc -f per_page=100

# 저장소 메타데이터가 부족해도 README에서 찾기
gh api -X GET search/code \
  -f q='"RTX 5090" filename:README.md' -f per_page=100

# 단순 하드웨어 언급 대신 측정 결과 찾기
gh search code '"RTX 5090" "tok/s"' --limit 100
gh search code '"RTX 5090" path:benchmarks' --limit 100
gh search code 'sm_120 path:results' --limit 100
gh search repos '"Claude Code" "RTX 5090"' --limit 100
```

### 다른 출처

- [Hugging Face 전체 텍스트 검색](https://huggingface.co/search/full-text?q=%22RTX%205090%22)으로 모델 카드, Space와 데이터셋을 찾을 수 있습니다. 예를 들어 [witcheer/rtx-5090-benchmarks](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks)는 양자화 LLM의 속도와 품질 결과를 공개합니다.
- arXiv와 학회 논문 전체 텍스트 검색으로 학습·평가 하드웨어를 찾을 수 있지만 5090을 사용했다는 사실만으로는 이 목록에 유용하지 않습니다. 공개 코드, 원시 결과와 정확한 설정이 있는 자료를 우선합니다.
- Reddit, NVIDIA Developer Forums와 제조사 포럼은 호환성 해결책을 발견하기에 좋습니다. 가능하면 포럼 글 자체보다 연결된 저장소나 재현 가능한 보고서를 등재합니다.

좋은 검색식은 하드웨어 식별자와 증거 키워드를 결합합니다. 예: `"RTX 5090" "tok/s"`, `"RTX 5090" VRAM`, `"RTX 5090" benchmark`, `sm_120 results`, `GB202 CUDA`. 항목을 추가하기 전에 GPU 변형, 소프트웨어 버전, 모델·정밀도, 작업 부하, 원시 출력, 실측인지 추정인지 확인합니다.

## 선정 기준

[GitHub `rtx-5090` 토픽](https://github.com/topics/rtx-5090?o=asc&s=forks)에는 2026-09-18 조사 시점에 공개 저장소 90개가 있었습니다. 이 목록은 GitHub 코드, 저장소 README, 결과 디렉터리와 Hugging Face도 함께 검색합니다. 토픽은 발견을 위한 단서일 뿐입니다. README에 다음 중 하나 이상이 있을 때 포함합니다.

- 측정 조건을 확인할 수 있는 재현 가능한 RTX 5090 결과
- Blackwell 소비자 GPU의 `sm_120`/`sm_120a` 전용 설정 또는 호환성 수정
- RTX 5090에서 명시적으로 시험한 응용 워크플로
- RTX 5090 보드나 전력 경로를 직접 지원하는 하드웨어 도구

카드 이름이 키워드 목록, 일반 하드웨어 목록, 홍보 문구 또는 무관한 암호화폐 홍보에만 나타나는 프로젝트는 제외합니다. 방치되었거나 안전하지 않거나 검증하기 어렵거나 기존 항목과 거의 같은 프로젝트도 생략할 수 있습니다. 목록 포함은 보증이 아닙니다. 코드, 라이선스, 모델 약관과 다운로드 파일을 직접 검토하세요.

## 기여

기여를 환영합니다. [CONTRIBUTING.md](CONTRIBUTING.md)를 읽고 pull request 하나에 프로젝트 하나를 제출해 주세요. 정확한 RTX 5090 증거를 적고 결과 또는 설정 설명의 permalink를 우선해 주세요.

## 라이선스

이 목록은 [CC0 1.0](LICENSE)으로 공개합니다. 링크된 프로젝트의 라이선스와 상표는 각 소유자에게 있습니다.
