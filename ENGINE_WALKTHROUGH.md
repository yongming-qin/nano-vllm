# Engine Walkthrough

This note is a guided map for `nanovllm/engine/` and the annotated copy in
`nanovllm/engine_annotated/`.

If you only want one reading order, use this:

1. `nanovllm/engine_annotated/llm_engine.py`
2. `nanovllm/engine_annotated/scheduler.py`
3. `nanovllm/engine_annotated/sequence.py`
4. `nanovllm/engine_annotated/block_manager.py`
5. `nanovllm/engine_annotated/model_runner.py`

## Big Picture

The engine is a small inference serving loop inspired by vLLM.

Its job is to:

- accept prompts
- tokenize them
- batch requests together
- manage a paged KV cache
- run the model for either prefill or decode
- sample next tokens
- stop requests when EOS or `max_tokens` is reached

The core design idea is that generation is split into two phases:

- `prefill`: process a prompt and write its KV states into cache
- `decode`: generate one new token per active request

This split matters because prompt processing is expensive and irregular, while
decode is small, repeated, and much easier to optimize.

## Main Objects

### `LLMEngine`

`LLMEngine` is the top-level coordinator.

It owns:

- the tokenizer
- the scheduler
- one local `ModelRunner`
- optional extra worker processes for tensor parallelism

Public flow:

1. `generate()` receives a list of prompts.
2. Each prompt becomes a `Sequence` via `add_request()`.
3. The scheduler decides the next batch to run.
4. `ModelRunner.run()` executes the batch.
5. The scheduler appends sampled tokens and marks finished requests.
6. `generate()` loops until all requests finish.

In other words, `LLMEngine` does not itself implement scheduling or model
execution details. It mostly glues the subsystems together.

### `Sequence`

`Sequence` is the mutable state for one request.

It tracks:

- all tokens so far
- how many tokens came from the original prompt
- the last generated token
- how many prompt tokens were reused from prefix cache
- the sequence's KV-cache block table
- sampling settings like temperature and max length

One useful way to think about it:

- `token_ids` is the logical text history
- `block_table` is where that history lives in physical KV-cache memory

### `Scheduler`

The scheduler decides which sequences run next.

It maintains two queues:

- `waiting`: requests not currently active on GPU
- `running`: requests already admitted and participating in decode

It has two modes:

- prefill mode: move requests from `waiting` to `running`
- decode mode: advance running requests by one token each

### `BlockManager`

This is the KV-cache allocator.

It splits cache memory into fixed-size blocks and hands those blocks to
sequences. It also implements prefix caching by hashing full token blocks.

This is the main vLLM-like idea in the repo.

### `ModelRunner`

This is the GPU execution backend.

It is responsible for:

- distributed initialization
- loading model weights
- allocating the KV cache
- preparing tensors for attention kernels
- running the forward pass
- sampling token ids on rank 0
- optionally using CUDA graphs for decode

## End-to-End Flow

### 1. User Calls `generate()`

You usually enter through `LLM.generate(...)`, which inherits from
`LLMEngine.generate(...)`.

For each prompt:

- strings are tokenized
- token ids are wrapped in a `Sequence`
- the sequence is appended to `scheduler.waiting`

At this point nothing has run on GPU yet. The requests are only queued.

### 2. The Engine Calls `step()`

Each loop iteration calls:

1. `scheduler.schedule()`
2. `model_runner.run(seqs, is_prefill)`
3. `scheduler.postprocess(seqs, token_ids)`

That is the entire engine heartbeat.

Everything interesting happens inside those three calls.

## Scheduling Logic

### Prefill Comes First

The scheduler first tries to admit waiting requests.

A request can be prefilling only if:

- the number of active sequences stays under `max_num_seqs`
- the token budget stays under `max_num_batched_tokens`
- enough KV-cache blocks are available

If one or more waiting requests fit, the scheduler returns them as a prefill
batch.

Why prefer prefill first?

Because new prompts can often be processed in parallel, and once admitted they
become part of the decode set.

### Decode Happens Otherwise

If no waiting request can be admitted, the scheduler switches to decode mode.

In decode:

- each scheduled sequence contributes exactly one next token
- before that token is produced, the scheduler checks whether the sequence can
  extend its KV-cache footprint

If cache space is insufficient, the scheduler may preempt another running
sequence. In this implementation, preemption means:

- free its blocks
- move it back to `waiting`
- later recompute it from its token history

That is simpler than swapping cache to CPU and is enough for an educational
repo.

## KV Cache and Blocks

### Why Blocks?

Instead of storing KV cache as one contiguous area per request, the cache is
divided into fixed-size blocks.

Benefits:

- requests can grow independently
- memory can be reused more flexibly
- identical prompt prefixes can share previously computed blocks

This mirrors the main paged-attention idea from vLLM.

### `block_table`

Each sequence has a `block_table`, which maps logical block index to physical
cache block id.

Example:

- logical tokens `0..255` may live in physical block `17`
- logical tokens `256..511` may live in physical block `3`

The model does not care about logical sequence layout directly. Attention code
uses `block_table` and related metadata to find the right cached KV entries.

### Prefix Cache

During allocation, `BlockManager.allocate()` walks the sequence block by block.

For every full block, it computes a hash based on:

- the current block's token ids
- the previous block's hash

This makes the hash depend on the full prefix, not only the current block.

If a matching hash already exists, and token ids also match, the block can be
reused.

That means:

- prompt tokens in reused blocks do not need to be recomputed
- `seq.num_cached_tokens` increases
- prefill only runs on the uncached suffix

Once a cache miss happens, later blocks are also treated as misses because the
prefix chain has changed.

### Appending During Decode

When a sequence generates new tokens, the block manager updates storage rules:

- if a new token starts a fresh block, allocate a new physical block
- if a trailing partial block just became full, hash it and add it to the
  prefix cache map
- otherwise, keep filling the current partial block

This is why `may_append()` looks a bit subtle: it handles transitions around
block boundaries.

## What Prefill Actually Sends to the Model

For prefill, `ModelRunner.prepare_prefill()` builds:

- `input_ids`: only the uncached suffix tokens
- `positions`: token positions for those suffix tokens
- `cu_seqlens_q`: cumulative lengths of query tokens
- `cu_seqlens_k`: cumulative lengths of key/value context
- `slot_mapping`: where new KV values should be written
- `block_tables`: where cached blocks already live, if prefix cache is used

The important mental model is:

- query length = tokens you are computing now
- key/value length = full context the attention can read

If a prefix is cached, key/value length is larger than query length.

## What Decode Actually Sends to the Model

For decode, `ModelRunner.prepare_decode()` is much simpler.

Each active sequence contributes:

- its latest token as `input_ids`
- the current last position
- its total context length
- the exact cache slot where the new KV entry should be written
- its padded `block_table`

Decode is regular: one token per sequence. That regularity is why CUDA graphs
work well for this phase.

## Why `Sequence` Has Custom Pickle Logic

With tensor parallelism, extra worker processes need request metadata.

But sending the full token history every decode step would be wasteful.

So `Sequence.__getstate__()` does something clever:

- before generation starts, it sends full prompt tokens
- after that, it only sends the latest token plus metadata

This works because old attention state already lives in the KV cache on each
GPU. Workers do not need the full decoded history every step.

## Tensor Parallel Flow

If `tensor_parallel_size > 1`:

- rank 0 lives in the main process
- ranks 1..N are spawned as child processes
- all ranks initialize an NCCL process group
- rank 0 writes method calls into shared memory
- worker ranks wake up, read the command, and run the same method locally

Rank 0 is special because it also performs sampling and returns token ids back
to the scheduler.

Conceptually:

- all ranks run the model forward
- only rank 0 decides which token was sampled

## KV Cache Allocation

`ModelRunner.allocate_kv_cache()` estimates how many blocks can fit in GPU
memory after model loading and warmup.

It computes:

- available memory budget
- bytes required per KV block
- number of blocks that fit

Then it allocates one large tensor for all layers and attaches per-layer slices
to attention modules.

This means the cache is not allocated block-by-block in Python objects. The
Python side only manages metadata. The actual KV storage is one big GPU tensor.

## CUDA Graph Path

If eager mode is disabled, `ModelRunner.capture_cudagraph()` pre-captures decode
graphs for common batch sizes.

Why only decode?

- prefill shapes vary too much
- decode has stable shapes and repeated structure

During runtime:

- pick the smallest captured graph that fits the current batch size
- copy current inputs and metadata into static staging tensors
- replay the graph

This reduces launch overhead in steady-state decoding.

## Stop Conditions

After each decode step, `scheduler.postprocess()` appends the sampled token to
each sequence.

A sequence finishes when either:

- the sampled token equals EOS and `ignore_eos` is false
- generated token count reaches `max_tokens`

When finished:

- its status becomes `FINISHED`
- its blocks are deallocated
- it is removed from `running`

## Recommended Reading Strategy

If you are learning the repo for the first time, use this order:

1. Read `nanovllm/engine_annotated/llm_engine.py` to understand the outer loop.
2. Read `nanovllm/engine_annotated/scheduler.py` to understand batching policy.
3. Read `nanovllm/engine_annotated/sequence.py` to understand request state.
4. Read `nanovllm/engine_annotated/block_manager.py` to understand paged cache
   and prefix reuse.
5. Read `nanovllm/engine_annotated/model_runner.py` last, because it contains
   the most low-level tensor plumbing.

If `model_runner.py` feels dense, that is normal. The essential thing to grasp
first is:

- scheduler chooses work
- block manager decides memory layout
- model runner turns that layout into tensors the model can execute

## One-Sentence Summary

The whole engine is a loop that repeatedly asks, "Which requests can run with
the current KV-cache budget?" and then executes either prompt prefill or
one-token decode for that chosen batch.
