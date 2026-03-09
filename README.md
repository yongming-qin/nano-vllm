
# My Note

My gpu is rtx 5090. `pip install flash-attn` installs the version which does not support rtx 50.
Then my vllm conda env is
torch 2.9.0 + cu128 + flash-attn 2.8.3 + python 3.12
Then the example.py will fail because the library file does not match.

Solution:
```
export TORCH_CUDA_ARCH_LIST="12.0"
MAX_JOBS=$(nproc) pip install flash-attn --no-build-isolation --no-binary :all:
```
The first line ensures we only build the support for rtx 50 gpus, which leads to shorter time (30 seconds for my case).


# Implemented Features


For an educational repo, it implements a surprisingly solid slice of a modern LLM inference engine.

The most noteworthy features are:

- **Paged KV cache with block tables.** Requests do not own one giant contiguous cache region; they use fixed-size blocks managed by `nanovllm/engine/block_manager.py`. That is the core vLLM-style memory idea.
- **Prefix caching.** Full prompt blocks are hashed and reused, so repeated prefixes can skip recomputation. You can see this in `nanovllm/engine/block_manager.py` and how prefill uses `block_tables` in `nanovllm/engine/model_runner.py`.
- **Separate batched prefill and batched decode.** The scheduler in `nanovllm/engine/scheduler.py` explicitly distinguishes prompt admission from one-token decode, which is how real serving systems are structured.
- **Preemption under KV-cache pressure.** If a running sequence cannot grow, the scheduler can evict another running sequence back to `waiting` and later recompute/re-admit it. That is a real serving-system concern, not just toy code.
- **Variable-length prefill without normal token padding.** `nanovllm/engine/model_runner.py` packs tokens and uses `cu_seqlens_q` / `cu_seqlens_k`, which is a more realistic inference layout than simple padded `[B, T]` tensors.
- **FlashAttention integration.** `nanovllm/layers/attention.py` uses `flash_attn_varlen_func` for prefill and `flash_attn_with_kvcache` for decode, so the repo is not just conceptually like vLLM, it uses the same class of optimized attention kernels.
- **Custom KV-cache write path.** `nanovllm/layers/attention.py` includes a Triton kernel to store keys/values into paged cache slots.
- **Tensor parallelism.** `nanovllm/engine/model_runner.py`, `nanovllm/layers/linear.py`, and `nanovllm/layers/embed_head.py` implement multi-rank execution with NCCL, sharded linear layers, sharded embeddings, and sharded LM head.
- **Decode CUDA graphs.** `nanovllm/engine/model_runner.py` captures CUDA graphs for common decode batch sizes, which is a real inference optimization.
- **`torch.compile` on hot ops.** `nanovllm/layers/sampler.py`, `nanovllm/layers/activation.py`, `nanovllm/layers/layernorm.py`, and `nanovllm/layers/rotary_embedding.py` compile frequently used kernels.
- **A full model implementation, not just an engine shell.** `nanovllm/models/qwen3.py` implements a Qwen3-style transformer with RoPE, RMSNorm, parallel QKV projections, MLP, and LM head.
- **Weight loading for packed/sharded weights.** `nanovllm/utils/loader.py` handles safetensors and packed module mappings like QKV and gate/up projections.
- **vLLM-like Python API and offline benchmark.** `nanovllm/llm.py`, `example.py`, and `bench.py` make it usable as a minimal inference library rather than only a code demo.

If I had to compress it to the most impressive three, I’d say:

1. `paged KV cache + prefix caching`
2. `batched prefill/decode scheduler`
3. `real performance features: FlashAttention + tensor parallel + CUDA graphs`

One useful limitation to keep in mind: this repo is still intentionally small, so it is closer to an **offline educational inference engine** than a full production server. For example, it appears focused on the `Qwen3` model path in `nanovllm/models/qwen3.py`, rather than supporting many architectures.


# Missing Features

This repo covers the **core inference engine ideas**, but it leaves out many features that matter in real `vLLM` or `SGLang` deployments.

The biggest missing areas are:

- **Chunked prefill.** This repo does whole-prompt prefill, then decode. Modern systems often split long prompts into chunks so long prefills do not block decode traffic.
- **Serving-oriented continuous batching policies.** This repo has a simple prefill-first scheduler in `nanovllm/engine/scheduler.py`, but production systems usually have more nuanced fairness, latency control, and admission policies.
- **HTTP / OpenAI-compatible server layer.** This repo is an offline Python API around `LLM.generate()`, not a full online serving stack with request streaming, cancellation, backpressure, and metrics.
- **Streaming tokens to clients.** Here you get results after completion, not token-by-token streaming over an API connection.
- **Broader model support.** This repo is centered on `Qwen3` in `nanovllm/models/qwen3.py`; real `vLLM` and `SGLang` support many architectures and variants.
- **LoRA / adapter serving.** A big real-world feature is loading base models once and serving many LoRA adapters dynamically.
- **Quantization support.** Production systems often support AWQ, GPTQ, FP8, INT8, and other quantized inference modes.
- **Speculative decoding.** This is a major speedup technique in modern serving systems and is not part of this repo’s engine.
- **Constrained / structured decoding.** Things like JSON schema guidance, regex/grammar constraints, function/tool calling support, and guided decoding are important in real APIs.
- **Multimodal support.** vLLM and SGLang both increasingly support vision-language style inputs; this repo is text-only.
- **More distributed execution modes.** This repo has tensor parallelism, but not the wider set of deployment patterns like pipeline parallelism, expert parallelism, disaggregated prefill/decode, or large multi-node scheduling.
- **Production-grade KV-cache policies.** It has paged KV cache and prefix caching, which is great, but not the richer cache eviction, reuse, and memory policies you’d expect in larger systems.
- **Request/session features.** Real systems often support session continuation, conversation state handling, prompt cache sharing across requests, and more advanced lifecycle management.
- **Observability and ops features.** Missing things include detailed metrics, tracing, profiling hooks, health checks, autoscaling integration, and failure handling.
- **Advanced sampling surface.** This repo has temperature sampling, but not the broader set of decoding controls you usually see, like top-k, top-p, min-p, repetition penalties, beam search, best-of, etc., unless implemented elsewhere beyond the core code we inspected.

A useful way to think about it is:

- This repo implements the **engine core**
- `vLLM` and `SGLang` also implement the **serving system around the engine**

So this repo already teaches you some of the hardest ideas:
- paged KV cache
- prefix caching
- batched prefill/decode
- tensor parallelism
- CUDA graph decode


<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>


# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)