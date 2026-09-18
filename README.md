# Awesome RTX 5090

**English** | [한국어](README.ko.md)

> A curated list of reproducible benchmarks, model recipes, inference engines, creative-AI workflows, and system tools tested on the NVIDIA GeForce RTX 5090.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub topic](https://img.shields.io/badge/GitHub%20topic-rtx--5090-181717?logo=github)](https://github.com/topics/rtx-5090)
[![License: CC0](https://img.shields.io/badge/license-CC0--1.0-blue.svg)](LICENSE)

Entries are selected for RTX 5090-specific measurements, setup instructions, or Blackwell `sm_120` support—not for merely mentioning the card.

> [!NOTE]
> Benchmark numbers are not directly comparable across rows unless the model, quantization, context, prompt/output lengths, backend, and sampling settings match. Follow each source link for the full test conditions.

## Contents

- [Quick reference](#quick-reference)
- [Model matrix](#model-matrix)
- [Benchmarks and evaluations](#benchmarks-and-evaluations)
- [LLM inference and serving](#llm-inference-and-serving)
- [Coding agents and developer tools](#coding-agents-and-developer-tools)
- [Windows and WSL](#windows-and-wsl)
- [Image, video, 3D, and audio](#image-video-3d-and-audio)
- [Computer vision](#computer-vision)
- [Scientific computing](#scientific-computing)
- [CUDA and ML libraries](#cuda-and-ml-libraries)
- [Hardware, power, and virtualization](#hardware-power-and-virtualization)
- [Selection policy](#selection-policy)
- [Contributing](#contributing)

## Quick reference

| Item | RTX 5090 desktop |
|---|---|
| Architecture | NVIDIA Blackwell, GB202 |
| Compute capability | `sm_120` / feature-specific `sm_120a` |
| VRAM | 32 GB GDDR7 |
| Common local-AI paths | Native Windows, WSL2, or Linux |
| Common inference formats | GGUF, FP8, NVFP4, BF16 |
| First places to look | [Broad benchmark suite](https://github.com/patrickwhelan-uk/llm-inference-benchmarks), [model quality evaluation](https://github.com/gisenberg/local-model-eval), [Windows guide](https://github.com/OnlyTerp/windows-is-fine-for-llms) |

Laptop RTX 5090 results are intentionally treated separately: the laptop GPU has a different memory capacity and power envelope, so desktop results should not be assumed to transfer.

## Model matrix

This table is an index into published results, not a leaderboard. “Reported result” reproduces the linked project's own measurement.

| Model | Quantization / precision | Backend | Reported RTX 5090 result | Context / workload | Source |
|---|---|---|---:|---|---|
| Ministral 3 3B | GGUF (source configuration) | LM Studio | 265.2 tok/s, 7.4 GB peak VRAM | 512-token prompt, 128 generated | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Llama 3.1 8B | FP16 | vLLM | 82 tok/s at concurrency 1; 304 tok/s aggregate at concurrency 4 | Short prompt, up to 256 output; smoke test | [llm-gpu-benchmark](https://github.com/DaytonLecoq/llm-gpu-benchmark#current-results-fp16-smoke-test) |
| Llama 3.1 8B | GGUF (source configuration) | LM Studio | 140.4 tok/s, 12.1 GB peak VRAM | 512-token prompt, 128 generated | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Qwen 2.5 7B | GGUF (source configuration) | LM Studio | 145.6 tok/s, 11.4 GB peak VRAM | 512-token prompt, 128 generated | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Qwen 2.5 Coder 32B | GGUF (source configuration) | LM Studio | 72.2 tok/s, 23.2 GB peak VRAM | 512-token prompt, 128 generated | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Qwen 3 8B | Q8_0 | imp | 385.4 tok/s | Single-stream decode; same GGUF/flags comparison | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Qwen 3 14B | Q6_K | imp | 162.5 tok/s | Single-stream decode; same GGUF/flags comparison | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Qwen 3 30B-A3B | Q4_K_M | imp | 305.5 tok/s | Single-stream decode; same GGUF/flags comparison | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Qwen 3.5 35B-A3B | Q4_K_M | Ollama | 145.6 tok/s peak; 120 tok/s at 131K | Windows 11; 196K max tested | [5090-windows11-benchmarks](https://github.com/taco-devs/5090-windows11-benchmarks/tree/master/qwen3.5-35b-a3b) |
| Qwen 3.6 35B-A3B | UD-Q4_K_M | imp | 287.9 tok/s | Single-stream decode; same GGUF/flags comparison | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Qwen 3.8 27B | NVFP4 | vLLM / NInfer | ~139 tok/s claimed by project | 262K configuration with MTP | [Qwen5090](https://github.com/Ark0N/Qwen5090) |
| Qwen 3.8 27B | NVFP4 weights + KV | vLLM | Turnkey serving recipe; no headline throughput | 262K, MTP-3, up to 8 streams | [vllm-sm120-nvfp4-mtp](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp) |
| Gemma 4 26B-A4B | Q6_K | llama.cpp / TurboQuant fork | 161.9 tok/s; 16/17 coding tests | Quality + throughput evaluation | [local-model-eval](https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_5090.md) |
| Gemma 4 31B | Q4_K_M | llama.cpp / TurboQuant fork | 50.3 tok/s; 17/17 coding tests | Full 262K reported at 28 GB | [local-model-eval](https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_5090.md) |
| GPT-OSS 20B | MXFP4 | imp | 382.7 tok/s | Single-stream decode; same model/flags comparison | [imp](https://github.com/kekzl/imp#how-fast-is-it-really) |
| Nemotron 3 Nano | GGUF (source configuration) | LM Studio | 237.5 tok/s, 26.9 GB peak VRAM | 512-token prompt, 128 generated | [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks#rtx-5090-lm-studio) |
| Nemotron 3 Nano 30B-A3B | INT4 / HQQ4 / k4v4 | Krasis | 151.76 tok/s internal decode | Hybrid GPU/CPU MoE runtime | [Krasis](https://github.com/brontoguana/krasis#benchmarks) |
| Nemotron 3 Super 120B-A12B | INT4 / HQQ4 / k4v4 | Krasis | 41.87 tok/s internal decode | Model larger than VRAM with expert residency | [Krasis](https://github.com/brontoguana/krasis#benchmarks) |
| Qwen 3.5 397B-A17B | INT4 / HQQ4 / k4v4 | Krasis | 10.04 tok/s internal decode | Model larger than VRAM with expert residency | [Krasis](https://github.com/brontoguana/krasis#benchmarks) |

Treat all numbers as snapshots of rapidly changing software. Rows marked “claimed by project” have useful setup detail but are not normalized or independently verified by this list.

## Benchmarks and evaluations

- [llm-inference-benchmarks](https://github.com/patrickwhelan-uk/llm-inference-benchmarks) — Reproducible LM Studio/llama.cpp harness covering decode, prompt evaluation, TTFT, VRAM, and power across many models.
- [local-model-eval](https://github.com/gisenberg/local-model-eval) — Tests coding quality alongside throughput, context capacity, and quantization trade-offs; includes raw experiment structure and 5090 rankings.
- [5090-windows11-benchmarks](https://github.com/taco-devs/5090-windows11-benchmarks) — Windows 11 context scaling, needle retrieval, VRAM limits, and Ollama/vLLM comparisons.
- [llm-gpu-benchmark](https://github.com/DaytonLecoq/llm-gpu-benchmark) — vLLM concurrency harness capturing latency percentiles, utilization, VRAM, temperature, and system metrics. The published result is currently a small FP16 smoke test.
- [NV-benchmark](https://github.com/QuanTuring-AI/NV-benchmark) — NIM/TensorRT-LLM versus Ollama study with throughput, TTFT, latency, and guardrail overhead data.
- [ecocompute-dynamic-eval](https://github.com/hongping-zh/ecocompute-dynamic-eval) — Compares model accuracy, cost, and energy/carbon behavior, including RTX 5090 quantization experiments.
- [gemma-rig](https://github.com/Stoneforge-Labs/gemma-rig) — Measured Gemma serving, evaluation, and agent-workflow results on one RTX 5090.
- [benchmark-rtx5090](https://github.com/pendakwahteknologi/benchmark-rtx5090) — Qwen 2.5 GGUF benchmark suite spanning four model sizes and three quantizations, with prompt/decode speed, power efficiency, and cost reports.
- [llm-bench](https://github.com/brenoperucchi/llm-bench) — Reproducible Ollama experiments covering PT/EN response quality, tool calling, concurrency, and long context, with raw results and a provenance manifest.
- [qwen3.8-27b-local-bench](https://github.com/Yunado/qwen3.8-27b-local-bench) — Qwen 3.8 27B evaluation across five GGUF checkpoints and ten KV-cache configurations using ARC-Challenge 500 and AIME 2026.

## LLM inference and serving

- [imp](https://github.com/kekzl/imp) — C++/CUDA inference engine tuned for `sm_120a`, with single-stream and concurrent benchmarks against llama.cpp and vLLM.
- [sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer) — CUDA inference runtime focused on MoE models and consumer/edge Blackwell GPUs; includes per-model RTX 5090 result files.
- [blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm) — SM120 patches and NVFP4 kernels for vLLM, FlashInfer, and CUTLASS.
- [vllm-nvfp4-docker](https://github.com/ckienstra/vllm-nvfp4-docker) — Docker build and serving stack for vLLM with NVFP4 KV-cache support on Blackwell.
- [vllm-sm120-nvfp4-mtp](https://github.com/seanyourhighness/vllm-sm120-nvfp4-mtp) — Pinned community vLLM image for Qwen 3.8 27B, NVFP4 KV, MTP-3, tool calls, and long context.
- [club-3090](https://github.com/noonghunna/club-3090) — Community serving recipes for Qwen and Gemma across one- and two-GPU 3090/4090/5090 systems.

## Coding agents and developer tools

- [qwen3.8-27b-claude-code-desktop-bridge](https://github.com/Maharajahu/qwen3.8-27b-claude-code-desktop-bridge) — Runs a local Qwen 3.8 27B Q6 GGUF through Claude Code Desktop or CLI on one RTX 5090. Its Anthropic-to-OpenAI bridge preserves reasoning, vision, streaming, and tool calls, and documents 200K text and 128K vision profiles.
- [Qwen5090](https://github.com/Ark0N/Qwen5090) — One-click Windows and scripted Linux setup for Qwen 3.8 27B with an OpenAI-compatible endpoint and local coding-agent clients, including Claude Code and DeepSeek Harness.
- [Krasis](https://github.com/brontoguana/krasis) — Hybrid Rust/CUDA runtime for serving MoE models far larger than VRAM. Published RTX 5090 results cover models from Nemotron 3 Nano 30B through Qwen 3.5 397B, with raw benchmark logs and quality checks.

## Windows and WSL

- [windows-is-fine-for-llms](https://github.com/OnlyTerp/windows-is-fine-for-llms) — Practical native-Windows versus WSL guidance, display-GPU TDR stability, memory limits, and llama.cpp serving.
- [comfyui-triton-and-sageattention-installer](https://github.com/DazzleML/comfyui-triton-and-sageattention-installer) — Cross-platform Triton and SageAttention installer with Windows and RTX 50-series support.
- [triton-blackwell-windows-pipeline](https://github.com/Arcen-c1c/triton-blackwell-windows-pipeline) — Windows build pipeline for Triton on `sm_120a`, including FP8 validation and benchmark outputs.

## Image, video, 3D, and audio

- [ComfyUI-Pixal3D](https://github.com/dreamrec/ComfyUI-Pixal3D) — ComfyUI integration for single-image-to-textured-PBR-mesh generation, tested on RTX 30/40/50 series.
- [local-ai-video-generator](https://github.com/carrotProgrammer/local-ai-video-generator) — Local ComfyUI video workflow using Wan 2.2 and MiniMax H3, verified on RTX 5090.
- [csm-rtx5090](https://github.com/D3velop-llc/csm-rtx5090) — CSM-1B streaming TTS pipeline optimized with CUDA graphs and `torch.compile` for Blackwell.
- [fish-s2-rtx](https://github.com/Genesis1231/fish-s2-rtx) — OpenAI-compatible OpenAudio S2-Pro streaming TTS and voice cloning on vLLM-Omni.
- [mpv-god-preset](https://github.com/GoldenSample/mpv-god-preset) — Measured mpv/VapourSynth setup for TensorRT upscaling, RIFE interpolation, HDR, and 4K/8K playback.
- [ComfyUI-Qlip](https://github.com/TheStageAI/ComfyUI-Qlip) — Compiled diffusion engines with RTX 5090 measurements for FLUX.2 Klein, Z-Image-Turbo, Wan 2.2, and other image/video models.
- [rtx509032g-minimax-h3-comfyui](https://github.com/Nuos/rtx509032g-minimax-h3-comfyui) — Reproducible MiniMax H3 text/image-to-video workflows with full wall-clock results, machine-readable data, and system-memory findings.
- [comfyui-vae-float32](https://github.com/AndreiOrehov/comfyui-vae-float32) — LTX video VAE precision and tiling experiments measured on an RTX 5090, including VRAM spill behavior.
- [h3-5090-cu130-benchmark](https://github.com/alangael24/h3-5090-cu130-benchmark) — Controlled MiniMax H3 profiling experiment comparing PyTorch/cu128 and PyTorch/cu130, with Nsight-derived per-kernel data, fixed workflow, and video output.

## Computer vision

- [traffic-vision-pipeline](https://github.com/GoktuGumus/traffic-vision-pipeline) — Tested YOLO vehicle detection, tracking, and line counting pipeline with stage-level timings and committed RTX 5090 benchmark JSON for batch and live-camera workloads.

## Scientific computing

- [one-gpu-n-qubits](https://github.com/drishans/one-gpu-n-qubits) — Reproducible quantum-circuit simulation experiments covering the VRAM wall, gate scaling, sampling, managed-memory offload, and cuTensorNet; every published number maps to provenance-carrying JSON.

## CUDA and ML libraries

- [FastFlashAttention](https://github.com/AlcAI-Haven/FastFlashAttention) — Exact BF16 flash attention with deterministic backward, tuned for Blackwell `sm_120`.
- [MinkowskiEngine](https://github.com/alpsaur/MinkowskiEngine) — Community fork updated for CUDA 12.8+, Blackwell, PyTorch 2.x, and modern NumPy; includes prebuilt wheels.
- [NATTEN Windows SM120 wheels](https://github.com/mmluankoko/NATTEN-0.21.6-Windows-Python3.13-Torch2.10-CUDA13.0-sm120) — Prebuilt Windows NATTEN wheel for Python 3.13, PyTorch 2.10, CUDA 13.0, and `sm_120`.
- [pytorch-tensorflow-gpu](https://github.com/dconsorte/pytorch-tensorflow-gpu) — Docker environment combining PyTorch and TensorFlow with CUDA 12.8 and Blackwell support. This repository is a fork; inspect its upstream delta before adopting it.

## Hardware, power, and virtualization

- [astral-hwmon](https://github.com/ksokolowski/astral-hwmon) — Linux `hwmon` sensors and guard tooling for per-pin 12VHPWR monitoring on ASUS ROG Astral cards.
- [ConnectorWatch](https://github.com/Shaderx/ConnectorWatch) — Experimental Windows input-voltage trend monitor with dashboard, tray notifications, and portable daemon.
- [gpu-tuner](https://github.com/xdzleo/gpu-tuner) — Searches NVIDIA RTX 40/50-series undervolt, overclock, and efficiency points.
- [vgpu-unlock-blackwell](https://github.com/bird/vgpu-unlock-blackwell) — Research into consumer Blackwell vGPU enablement. The README documents a current firmware/hardware blocker; this is research, not a working unlock.
- [macuda](https://github.com/Davinchy/macuda) — Experimental userspace NVIDIA driver and CUDA/cuBLAS shims running an RTX 5090 as a Thunderbolt eGPU on Apple Silicon macOS, with correctness gates and native comparisons.

## Selection policy

The [GitHub `rtx-5090` topic](https://github.com/topics/rtx-5090?o=asc&s=forks) contained **90 public repositories when reviewed on 2026-09-18**. Topic membership is only a discovery signal. A project is included when its README provides at least one of:

- reproducible RTX 5090 measurements and the relevant test conditions;
- a setup or compatibility fix specific to Blackwell consumer GPUs (`sm_120`/`sm_120a`);
- an application workflow explicitly tested on an RTX 5090;
- hardware tooling that directly supports an RTX 5090 board or power path.

Projects are excluded when the card appears only in a keyword list, generic hardware inventory, marketing copy, or unrelated cryptocurrency promotion. Abandoned, unsafe, unverifiable, or near-duplicate entries may also be omitted. Inclusion is not an endorsement; review code, licenses, model terms, and downloaded artifacts yourself.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and submit one project per pull request. Include the exact RTX 5090 evidence and prefer a permalink to results or setup instructions.

## License

This list is released under [CC0 1.0](LICENSE). Linked projects retain their own licenses and trademarks.
