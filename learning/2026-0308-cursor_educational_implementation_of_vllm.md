# Educational implementation of vllm
_Exported on 3/8/2026 at 22:37:40 PDT from Cursor (2.4.31)_

---

**User**

I want to learn this repo which is a mini implementation of vllm for education. Create a copy folder of @nanovllm/engine/ and add detailed comments in the files of engine folder.

---

**Cursor**

I’m going to inspect `nanovllm/engine/` first so I can create a commented learning copy that preserves the original implementation and explains each file clearly.

I’ve mapped the engine pieces; next I’m checking a couple of supporting types so the comments explain the real data flow instead of just restating the code.

I have enough context now. I’m going to create a separate `nanovllm/engine_annotated/` package that mirrors `engine/`, keeps the logic intact, and adds learning-focused docstrings plus inline comments so you can study the scheduling, cache, and runner flow without touching the original files.

I’m writing the annotated copy now. I’m keeping the code behavior the same and only adding explanatory structure so the files stay useful as a runnable reference.

The annotated package is mostly in place. I’m finishing the last engine file and then I’ll run a quick lint pass on the new package to catch any mistakes introduced while copying and commenting.

Created a separate annotated copy at `nanovllm/engine_annotated/` and left the original `nanovllm/engine/` untouched.

The new package mirrors the engine structure:
- `llm_engine.py`: request lifecycle, worker spawning, generation loop
- `scheduler.py`: prefill vs decode scheduling, preemption, finish handling
- `block_manager.py`: paged KV-cache blocks, prefix-cache hashing, allocation rules
- `sequence.py`: per-request state, block math, compact serialization
- `model_runner.py`: distributed setup, KV-cache allocation, prefill/decode tensor prep, CUDA graph path
- `__init__.py`: package entrypoint for the annotated copy

I added learning-focused module docstrings, class/method docstrings, and inline comments explaining why the code does things, not just what each line does. I also rewired the copied modules to import each other from `nanovllm.engine_annotated` so the copy is self-contained as a study package.

Verification: `ReadLints` reports no linter errors in `nanovllm/engine_annotated/`.

If you want, I can next add a short `ENGINE_WALKTHROUGH.md` that explains the end-to-end flow from `generate()` to sampling in one place.

---

**User**

yes

---

**Cursor**

I’m adding a concise walkthrough doc now that ties the annotated engine files together into one end-to-end story, from `generate()` through scheduling, KV-cache management, and sampling.

Added `ENGINE_WALKTHROUGH.md` at the repo root.

It explains the engine as a system rather than file-by-file comments: the main loop, `prefill` vs `decode`, scheduler policy, paged KV cache, prefix caching, tensor parallel workers, KV allocation, CUDA graphs, and a suggested reading order through `nanovllm/engine_annotated/`.

If you want, I can next add a small diagram to that doc showing the flow:
`generate() -> scheduler -> model_runner -> sampler -> postprocess`.

---

**User**

git commit and push

---

**Cursor**

I’m checking the repo state and recent commit style first, then I’ll commit only the annotated engine copy and walkthrough doc, leaving the untracked model folder and notebook out.

I’ve confirmed the only changes I’m committing are `nanovllm/engine_annotated/` and `ENGINE_WALKTHROUGH.md`; next I’m creating a commit in the repo’s existing style and then I’ll push `main` to `origin`.

The commit is in place and the branch is one commit ahead. I’m pushing now; the untracked `Qwen3-0.6B/` folder and `try.ipynb` remain local and were not included.

Committed and pushed to `origin/main`.

Commit: `4857e22`  
Message: `add annotated engine walkthrough`

Included:
- `nanovllm/engine_annotated/`
- `ENGINE_WALKTHROUGH.md`

Not included:
- `Qwen3-0.6B/`
- `try.ipynb`

If you want, I can also add a tiny example entrypoint that imports from `engine_annotated` so you can step through the annotated version directly in a debugger.

---

**User**

@nanovllm/engine/llm_engine.py:1-6 explain the imports

---

**Cursor**

In `nanovllm/engine/llm_engine.py`, those imports set up the engine’s top-level orchestration layer:

- `import atexit`: registers a cleanup hook so worker processes and distributed resources are shut down when Python exits. In this file, it is used to call `self.exit()` automatically.
- `from dataclasses import fields`: reads the field names from the `Config` dataclass so `LLMEngine` can filter `**kwargs` and pass only valid config options into `Config(...)`.
- `from time import perf_counter`: provides a high-resolution timer used to measure each prefill/decode step and compute throughput.
- `from tqdm.auto import tqdm`: creates the progress bar shown during `generate()`, including the live `Prefill` and `Decode` tok/s stats.
- `from transformers import AutoTokenizer`: loads the Hugging Face tokenizer for the selected model so string prompts can be converted to token ids and final token ids can be decoded back to text.
- `import torch.multiprocessing as mp`: spawns extra worker processes when `tensor_parallel_size > 1`, so multiple GPU ranks can run the model together.

So the pattern is: cleanup (`atexit`), config introspection (`fields`), timing (`perf_counter`), UX/progress (`tqdm`), text-token conversion (`AutoTokenizer`), and multi-process execution (`torch.multiprocessing`).

If you want, I can walk through the next block in `LLMEngine.__init__()` line by line.

---

**User**

@nanovllm/engine/llm_engine.py:48-54 explain line by line

---

**Cursor**

Here’s `step()` line by line:

```48:54:nanovllm/engine/llm_engine.py
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens
```

- `def step(self):`
  Defines one engine iteration. A single call to `step()` does not finish generation for all requests. It performs exactly one scheduling/execution/postprocess cycle.

- `seqs, is_prefill = self.scheduler.schedule()`
  Asks the scheduler what work should run next.
  `seqs` is the batch of `Sequence` objects chosen for this iteration.
  `is_prefill` tells you which phase this batch belongs to:
  - `True`: prompt prefill
  - `False`: decode

- `token_ids = self.model_runner.call("run", seqs, is_prefill)`
  Sends that batch to `ModelRunner`.
  It invokes the `run` method, passing both the sequences and whether this is prefill or decode.
  The return value `token_ids` is the sampled next token for each sequence in `seqs`.
  In decode, that is naturally “one token per sequence.”
  In prefill, the model processes the prompt and still returns one sampled next token at the end of each sequence.

- `self.scheduler.postprocess(seqs, token_ids)`
  Hands the sampled tokens back to the scheduler so it can update sequence state.
  This method:
  - appends each sampled token to its sequence
  - checks stopping conditions like EOS or `max_tokens`
  - marks finished sequences
  - deallocates KV-cache blocks for finished ones
  - removes finished ones from the running queue

- `outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]`
  Collects only the sequences that became finished after this step.
  For each finished sequence, it returns:
  - `seq.seq_id`: the request id
  - `seq.completion_token_ids`: only the generated continuation, not the original prompt

  This means unfinished sequences stay internal and are not returned yet.

- `num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)`
  Computes a metric used by `generate()` for throughput display.
  It uses a sign trick:
  - during prefill: positive number = total prompt-side tokens processed
  - during decode: negative number = number of sequences advanced

  Why `-len(seqs)` in decode?
  Because one decode step produces one token per sequence, so decode throughput is basically “how many sequences got one token this round.” The negative sign is just a compact way for the caller to distinguish decode from prefill.

- `return outputs, num_tokens`
  Returns:
  - `outputs`: newly finished requests from this step
  - `num_tokens`: workload info for progress/throughput reporting

In short, `step()` is the engine heartbeat:
1. schedule a batch
2. run the model
3. update sequence state
4. return any completed outputs

If you want, I can next explain why prefill also returns one sampled token even though it processes the whole prompt.

---

**User**

I don't understand the what scheduler does. @nanovllm/engine/scheduler.py:24-41 
explain line by line

---

**Cursor**

The scheduler’s job is to decide **which requests run next** and **whether this engine step is a prompt-processing step (`prefill`) or a one-token generation step (`decode`)**.

This block is the **prefill part** of scheduling:

```24:41:nanovllm/engine/scheduler.py
def schedule(self) -> tuple[list[Sequence], bool]:
    # prefill
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
            break
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True
```

Line by line:

- `def schedule(self) -> tuple[list[Sequence], bool]:`
  Defines the method that picks the next batch to run.
  It returns:
  - a list of sequences to execute now
  - a boolean saying whether this batch is `prefill` (`True`) or `decode` (`False`)

- `# prefill`
  This comment marks the first phase of scheduling: try to admit new requests from the waiting queue.

- `scheduled_seqs = []`
  Creates an empty list to hold the sequences selected for this batch.

- `num_seqs = 0`
  Tracks how many sequences have been added to this batch so far.

- `num_batched_tokens = 0`
  Tracks how many prompt tokens are in this prefill batch.
  This matters because the engine has a maximum token budget per batch.

- `while self.waiting and num_seqs < self.max_num_seqs:`
  Keep trying to admit requests as long as:
  - there are still waiting requests
  - the batch has not reached the maximum number of sequences

- `seq = self.waiting[0]`
  Look at the **next waiting request**, but do not remove it yet.
  The scheduler first checks whether it fits.

- `if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):`
  Stop if this request cannot be admitted.
  Two reasons:
  - adding its prompt would exceed the maximum batched token limit
  - there are not enough KV-cache blocks available for it

  `len(seq)` here means the current number of tokens in that request, which at prefill time is basically the prompt length.

- `break`
  If the next waiting request does not fit, stop admitting more requests for this prefill batch.

  Important detail: this scheduler is queue-ordered. It does **not** skip a large request and try later smaller ones. It stops at the front request.

- `num_seqs += 1`
  Count this request as part of the batch.

- `self.block_manager.allocate(seq)`
  Reserve KV-cache blocks for this sequence.
  This may also reuse prefix-cached blocks if part of the prompt was seen before.

- `num_batched_tokens += len(seq) - seq.num_cached_tokens`
  Increase the token budget by the number of tokens that actually need compute.

  This is subtle and important:
  - `len(seq)` = full prompt length
  - `seq.num_cached_tokens` = prompt tokens already available from prefix cache
  - so only the **uncached suffix** needs prefill computation

- `seq.status = SequenceStatus.RUNNING`
  Mark the request as active.
  It is no longer just waiting to be admitted.

- `self.waiting.popleft()`
  Remove it from the waiting queue.

- `self.running.append(seq)`
  Put it into the running queue, meaning it now owns KV-cache blocks and can participate in future decode steps.

- `scheduled_seqs.append(seq)`
  Add it to the list that will be returned for execution in this step.

- `if scheduled_seqs:`
  After the loop, check whether at least one waiting request was admitted.

- `return scheduled_seqs, True`
  If yes, return those sequences and mark this engine step as a **prefill** step.

So the big idea of this block is:

1. Take requests from `waiting` in order.
2. Admit as many as fit under:
   - max sequence count
   - max token budget
   - KV-cache capacity
3. Move them to `running`.
4. Return them as a **prefill batch**.

If you want, I can next explain the second half of `schedule()` at `scheduler.py:43-58`, which is the decode path and usually makes the scheduler’s full behavior click.

---

**User**

@nanovllm/engine/scheduler.py:5 what does BlockManager do in scheduler?

---

**Cursor**

`BlockManager` is the scheduler’s **memory manager for the KV cache**.

The scheduler decides **which sequence should run**.  
`BlockManager` tells it **whether there is enough KV-cache space to let that sequence run**, and updates the block bookkeeping when sequences start, grow, or finish.

In `scheduler.py`, it is used in four main ways:

- `self.block_manager = BlockManager(...)`
  Creates the KV-cache allocator using the configured number of blocks and block size.

- `self.block_manager.can_allocate(seq)`
  Used during **prefill**.
  It checks whether a waiting sequence can be admitted at all, meaning: are there enough free cache blocks for its current prompt?

- `self.block_manager.allocate(seq)`
  Also during **prefill**.
  It actually assigns physical KV-cache blocks to the sequence.
  This is also where prefix-cache reuse can happen if some prompt blocks were seen before.

- `self.block_manager.can_append(seq)`
  Used during **decode**.
  It checks whether the sequence has enough space to generate one more token. If not, the scheduler may preempt another sequence.

- `self.block_manager.may_append(seq)`
  Called right before/around decode growth bookkeeping.
  It updates block state when a sequence is about to extend into a new block or when a partial block becomes full.

- `self.block_manager.deallocate(seq)`
  Used when a sequence finishes or gets preempted.
  It frees the blocks so other sequences can use them.

So the simplest mental model is:

- `Scheduler` = traffic controller
- `BlockManager` = parking lot manager for KV-cache blocks

The scheduler cannot just say “run this request” unless `BlockManager` confirms there is room in cache. Without it, the scheduler would have no idea whether a request can fit on GPU memory or how its cached prefix is laid out.

If you want, I can explain `BlockManager.allocate()` next, since that is the most important method for understanding the vLLM idea.

---

**User**

before that, I wonder does this scheduler use batched prefill or batched decode? and 

It checks whether the sequence has enough space to generate one more token. If not, the scheduler may preempt another sequence.

what does preempt another sequence mean?

---

**Cursor**

Yes, this scheduler does **both batched prefill and batched decode**.

```24:58:nanovllm/engine/scheduler.py
def schedule(self) -> tuple[list[Sequence], bool]:
    # prefill
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
            break
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True

    # decode
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

### Batched prefill
Yes.

In the first `while` loop, it keeps taking requests from `self.waiting` and adding them to `scheduled_seqs` until one of these limits is hit:

- too many sequences: `num_seqs < self.max_num_seqs`
- too many prompt tokens: `max_num_batched_tokens`
- not enough KV-cache blocks: `can_allocate(seq)`

So prefill is **batched across multiple requests**.

### Batched decode
Also yes.

In the second `while` loop, it keeps taking sequences from `self.running` and adding them to `scheduled_seqs`. Each of those sequences will advance by **one token** in that engine step.

So decode is also batched, but in a different way:

- prefill batch: multiple sequences, many prompt tokens each
- decode batch: multiple sequences, one next token each

That is the standard serving pattern: **batched prompt processing + batched one-token decoding**.

## What “preempt another sequence” means

Here, **preempt** means:

- pick a currently running sequence
- stop letting it use KV-cache for now
- free its cache blocks
- move it back to the waiting queue

The code is here:

```60:63:nanovllm/engine/scheduler.py
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

So preemption in this repo does **not** mean killing the request or losing its text history. It means:

- the request still exists
- its `token_ids` are still in the `Sequence`
- but its GPU KV-cache state is discarded
- later, when space is available, it can be scheduled again and recomputed

This is a **recomputation-style preemption**.

### Why preempt?
Because decode may need one more block to keep growing a sequence. If there is no free block, the scheduler has to make room somehow. Its strategy is:

- evict another running sequence from GPU cache
- free those blocks
- let the current sequence continue

### Simple example
Imagine GPU KV cache can hold only 2 active sequences, but 3 are running or want to run:

- `A` is running
- `B` is running
- `C` wants space, or `A` needs another block for one more token

If there is no free block, the scheduler may preempt `B`:

- `B` is moved from `running` back to `waiting`
- `B`’s KV blocks are freed
- now `A` or `C` can use that memory

Later, `B` can come back, but it may need to rebuild its KV state by prefill/recomputation.

## Short version
- Yes, it uses **batched prefill**.
- Yes, it uses **batched decode**.
- “Preempt another sequence” means **temporarily evict one running request from KV-cache, free its blocks, and put it back in the waiting queue so another request can make progress**.

If you want, I can next explain why decode only produces **one token per sequence per step**, which is the key reason batching works this way.

---

**User**

@nanovllm/engine/scheduler.py:40-41 it seems if scheduled_seq include anything, then prefill will be triggered. Does this mean this logic prioritize prefill over decode?

---

**Cursor**

Yes, exactly: this logic **prioritizes prefill over decode**.

```40:41:nanovllm/engine/scheduler.py
if scheduled_seqs:
    return scheduled_seqs, True
```

Here’s what happens in `schedule()`:

1. It first tries the **prefill path** by pulling from `self.waiting`.
2. If it successfully schedules **at least one** waiting request, `scheduled_seqs` is non-empty.
3. It immediately returns `(..., True)`, meaning:
   - run this batch as **prefill**
   - do **not** continue to the decode logic in this same step

So yes, the control flow is:

- try prefill first
- only if **no prefill batch can be formed** does it fall through to decode

That means decode is a fallback, not the first choice.

## Why this means “prefill priority”
Because the function has this structure:

```python
try prefill
if any prefill work exists:
    return prefill
otherwise:
    do decode
```

So in one engine step, it never mixes:

- some prefill sequences
- some decode sequences

It chooses **one mode per step**, and prefill gets first chance.

## More precise nuance
It does **not** mean “always prefill whenever waiting is non-empty.”

It means “prefill whenever at least one waiting sequence can actually be admitted.”

A waiting sequence might fail prefill because of:

- `max_num_batched_tokens`
- `max_num_seqs`
- insufficient KV-cache blocks via `can_allocate(seq)`

If the first waiting request cannot be admitted, then prefill returns nothing, and the scheduler moves on to decode.

## Why might the author do this?
A likely reason is simplicity and throughput:

- admitting new prompts early keeps the system from ignoring queued requests
- prefill can batch many prompt tokens together efficiently
- separating prefill and decode makes the scheduler and runner easier to reason about

In larger production systems, the tradeoff is more subtle, because too much prefill priority can hurt decode latency for already-running requests. But in this educational repo, this design is simple and clean.

## One-sentence takeaway
Yes, `scheduler.py:40-41` means the scheduler is **prefill-first**: if it can form any prefill batch at all, it returns that immediately and skips decode for this step.

---

**User**

I heard for batched prefill, the sequences' length are different. does this code do something about this? like padding? or what if one finished and others are not finished in prefill?

---

**Cursor**

I’m checking the prefill preparation path in `model_runner.py` because that’s where variable-length sequences and “one finishes, others continue” behavior are handled.

Yes. This code **does handle different sequence lengths in batched prefill**, and it mostly does **not** do classic token padding.

The key logic is here:

```126:162:nanovllm/engine/model_runner.py
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None
    for seq in seqs:
        seqlen = len(seq)
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))
        seqlen_q = seqlen - seq.num_cached_tokens
        seqlen_k = seqlen
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        ...
```

## Different lengths: how it works

Suppose you batch 3 prompts with lengths `5`, `10`, and `20`.

This code does **not** make a dense `[batch, max_len]` token matrix like:

```python
[
  [a b c d e PAD PAD ...]
  [a b c d e f g h i j ...]
  ...
]
```

Instead, it does something closer to:

- concatenate all real tokens into one long `input_ids`
- record boundaries using `cu_seqlens_q` and `cu_seqlens_k`

So it is using a **packed / ragged representation**, not standard padding.

### `input_ids.extend(...)`
Each sequence contributes only its real tokens:

```137:143:nanovllm/engine/model_runner.py
input_ids.extend(seq[seq.num_cached_tokens:])
positions.extend(list(range(seq.num_cached_tokens, seqlen)))
seqlen_q = seqlen - seq.num_cached_tokens
seqlen_k = seqlen
cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
max_seqlen_q = max(seqlen_q, max_seqlen_q)
max_seqlen_k = max(seqlen_k, max_seqlen_k)
```

That means:

- no per-token padding for `input_ids`
- no per-token padding for `positions`
- lengths are tracked by cumulative sequence-length arrays

### What are `cu_seqlens_q` and `cu_seqlens_k`?
They tell the attention backend where each sequence begins and ends in the flattened token buffer.

For example, if uncached query lengths were `[5, 10, 20]`, then:

- `cu_seqlens_q = [0, 5, 15, 35]`

That means:

- sequence 1 uses `input_ids[0:5]`
- sequence 2 uses `input_ids[5:15]`
- sequence 3 uses `input_ids[15:35]`

So **variable lengths are handled explicitly**, not with token padding.

## Is anything padded at all?
Yes, but not the main prompt tokens.

The part that does use padding is the `block_tables` tensor:

```120:124:nanovllm/engine/model_runner.py
def prepare_block_tables(self, seqs: list[Sequence]):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    return block_tables
```

So:

- `block_table` lengths can differ by sequence
- they are padded with `-1` to the same width

But that is **metadata padding**, not prompt-token padding.

## What if one sequence finishes and others do not finish in prefill?
That is handled cleanly.

After prefill runs, the engine samples **one next token per sequence**, then `postprocess()` updates each sequence independently:

```65:71:nanovllm/engine/scheduler.py
def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

So after a batched prefill step:

- sequence A may sample EOS and finish immediately
- sequence B may continue running
- sequence C may continue running

They do **not** have to all finish together.

### Important nuance
A sequence does not “finish halfway through prefill” in the sense of stopping in the middle of prompt processing. Prefill is just processing the prompt and then sampling the next token. The finish check happens **after** the sampled token is produced.

So the unit of work is:

1. process the whole prompt for that sequence
2. sample one next token
3. decide if that sequence is finished

## Short answer
- Yes, batched prefill supports **different sequence lengths**.
- It does this with **packed tokens + cumulative lengths**, not normal token padding.
- The only obvious padding here is for `block_tables`, padded with `-1`.
- If one sequence finishes after prefill and others do not, `postprocess()` marks only that sequence finished and leaves the others running.

If you want, I can draw a tiny concrete example with two prompts, showing exactly what `input_ids`, `cu_seqlens_q`, and `cu_seqlens_k` look like.

---

**User**

you mentioned
It does this with packed tokens + cumulative lengths, not normal token padding.

so for different sequence lengths batched prefill, there are other solutions, right? what are other solutions?

---

**Cursor**

Yes. **Packed tokens + cumulative lengths** is one solution, but not the only one.

For batched prefill with different sequence lengths, common approaches are:

## 1. Dense padding to max length
This is the most familiar approach.

You make a dense tensor of shape roughly `[batch_size, max_seq_len]`, and shorter sequences are padded.

Example:

```python
[
  [11, 22, 33,  0,  0],
  [44, 55, 66, 77, 88],
]
```

Then you use an attention mask so the model ignores padding tokens.

### Pros
- simple to understand
- easy to implement
- works with standard transformer code

### Cons
- wastes compute on padding
- wastes memory bandwidth
- inefficient when sequence lengths vary a lot

This is common in training and in simpler inference systems.

## 2. Packed / ragged representation
This is what this repo is doing for prefill.

Instead of padding, you flatten all real tokens into one 1D buffer and keep boundary metadata like `cu_seqlens`.

### Pros
- avoids wasted compute on pad tokens
- better for variable-length batches
- often used with optimized attention kernels like FlashAttention variants

### Cons
- more complex bookkeeping
- requires kernels that understand packed layout

This is usually better for serving systems.

## 3. Bucketed batching
Another strategy is to group requests with similar lengths together before batching.

For example:

- batch A: prompts of length 1-32
- batch B: prompts of length 33-64
- batch C: prompts of length 65-128

Then you may still pad inside each bucket, but padding waste is smaller.

### Pros
- simpler than full ragged kernels
- less padding waste than naive batching
- easy scheduling tradeoff

### Cons
- still wastes some compute
- batching becomes more constrained
- can increase waiting time if you wait for the “right” bucket

A lot of systems combine bucketing with padding.

## 4. One request per prefill
The simplest possible design: no batching for prefill at all.

Each prompt is run alone.

### Pros
- easiest logic
- no variable-length batching issues

### Cons
- poor throughput
- underuses GPU badly

This is mostly for toy systems or debugging.

## 5. Chunked prefill
Long prompts can be split into chunks and processed across multiple steps instead of one huge prefill.

For example, a 4000-token prompt might be prefed as:

- chunk 1: tokens 0-1023
- chunk 2: tokens 1024-2047
- chunk 3: tokens 2048-3071
- chunk 4: tokens 3072-3999

This can be mixed with batching.

### Pros
- avoids giant one-shot prefills
- helps latency fairness
- can interleave with decode work

### Cons
- more scheduler complexity
- more state bookkeeping
- potentially less straightforward than single-shot prefill

Modern serving systems often use some form of chunked prefill for latency control.

## 6. Paged / block-based context handling
This is slightly different from token batching, but related.

Instead of requiring sequence memory to be contiguous, systems store KV cache in blocks/pages and use block tables to map logical sequence positions to physical memory.

That is what this repo’s `BlockManager` is about.

This does not itself solve “how to batch variable-length prompt tokens,” but it helps a lot with:
- memory management
- prefix cache reuse
- dynamic sequence growth

So in practice, production systems often combine:
- paged KV cache
- packed or chunked prefill
- batched decode

## 7. Specialized attention kernels with sequence metadata
Some kernels support variable-length sequences directly using metadata like:
- cumulative sequence lengths
- per-sequence lengths
- block tables
- slot mappings

This is basically the kernel-level version of the packed approach.

Instead of adapting the model to dense padded tensors, the attention kernel understands irregular layout directly.

### Pros
- high efficiency
- avoids unnecessary padding compute
- good fit for serving

### Cons
- harder to implement
- more tightly coupled to custom kernels/runtime

## What this repo chose
This repo’s prefill path is closest to:

- **packed/ragged tokens**
- plus **cumulative sequence lengths**
- plus **paged KV-cache metadata**
- plus optional **block tables** when prefix cache is used

So it avoids classic `[batch, max_len]` prompt padding.

## Simple comparison
If prompt lengths are `[5, 1000]`:

- dense padding computes something like `2 x 1000`
- packed representation computes only `5 + 1000 = 1005` real tokens

That is why packed approaches are attractive for inference serving.

## Practical summary
The main alternatives are:

- padding + mask
- bucketing + padding
- packed/ragged layout
- chunked prefill
- no batching
- combinations of the above

For educational clarity:
- training code often uses **padding**
- efficient inference servers often use **packed or chunked approaches**
- vLLM-style systems combine that with **paged KV cache**

If you want, I can next compare **packed prefill vs padded prefill** using a tiny concrete tensor example with shapes and arrays.

---

**User**

@nanovllm/engine/scheduler.py:43-58 let's come back to scheduler.py, what does these lines do?

---

**Cursor**

This block is the **decode scheduling path**. It runs only if the scheduler could not form a prefill batch.

Its job is:

- choose which already-running sequences get to generate **one more token**
- make sure there is KV-cache space for that next token
- preempt other sequences if necessary

Here is the code again:

```43:58:nanovllm/engine/scheduler.py
# decode
while self.running and num_seqs < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())
        else:
            self.preempt(seq)
            break
    else:
        num_seqs += 1
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))
return scheduled_seqs, False
```

## Big picture first

At this point:

- `self.running` contains sequences already active on GPU
- each decode step gives each scheduled sequence **one next token**
- but before a sequence can grow, the scheduler checks whether there is enough KV-cache space

If not enough space exists, it may **preempt** another sequence to free blocks.

## Line by line

- `# decode`
  Marks the decode phase.

- `while self.running and num_seqs < self.max_num_seqs:`
  Keep selecting running sequences as long as:
  - there are running sequences available
  - this decode batch has not hit the max sequence count

- `seq = self.running.popleft()`
  Take the next running sequence from the left side of the queue.
  This is the candidate sequence we want to let advance by one token.

- `while not self.block_manager.can_append(seq):`
  Before allowing `seq` to decode one more token, check whether it has enough KV-cache space to grow.

  If `can_append(seq)` is `False`, that means this sequence cannot safely continue right now without freeing some blocks first.

- `if self.running:`
  If there are still other running sequences available, the scheduler can evict one of them to make room.

- `self.preempt(self.running.pop())`
  Remove one sequence from the **end** of the running queue and preempt it.
  That means:
  - mark it `WAITING`
  - deallocate its KV-cache blocks
  - move it back to the waiting queue

  So the scheduler sacrifices another running sequence to help the current `seq` continue.

- `else:`
  This means there are no other running sequences left to evict.

- `self.preempt(seq)`
  If `seq` itself still cannot append and no one else can be evicted, then preempt `seq` too.

- `break`
  Stop trying to schedule this `seq`, because it could not be made runnable.

## Important Python detail: the `while ... else`
This part is easy to miss:

```python
while not self.block_manager.can_append(seq):
    ...
else:
    ...
```

The `else` runs only if the `while` loop finishes **without hitting `break`**.

So the meaning is:

- if the scheduler eventually makes enough room for `seq`, run the `else` block
- if it had to `break` because `seq` itself got preempted, skip the `else` block

## Continue line by line

- `num_seqs += 1`
  Count this sequence as part of the decode batch.

- `self.block_manager.may_append(seq)`
  Update KV-cache bookkeeping for the fact that `seq` is about to advance by one token.

  This may:
  - allocate a new block if decoding enters a fresh block
  - finalize/hash a block if it just became full
  - or do nothing special if still inside the same partial block

- `scheduled_seqs.append(seq)`
  Add this sequence to the decode batch that will be returned.

- `assert scheduled_seqs`
  Sanity check: by the time decode path finishes, at least one sequence should have been scheduled.

- `self.running.extendleft(reversed(scheduled_seqs))`
  Put the scheduled sequences back at the front of the running queue, preserving their original order.

  Why?
  Because they are still active after this step unless they later finish in `postprocess()`.

- `return scheduled_seqs, False`
  Return the chosen decode batch.
  `False` means this is a **decode** step, not a prefill step.

## Intuition with a small example

Suppose `running = [A, B, C]`.

The scheduler starts decode:

1. pop `A`
2. check whether `A` can append one more token
3. if yes:
   - reserve/update block bookkeeping
   - schedule `A`
4. then try `B`
5. if `B` cannot append:
   - maybe preempt `C` to free space
   - check `B` again
6. if `B` can now append:
   - schedule `B`
7. after batching, put scheduled sequences back in front of `running`

So this is basically:  
“Try to let current running sequences continue decoding, but if memory is tight, evict some others.”

## What this block is really doing conceptually

It is implementing **fair-ish batched decode under a limited KV-cache budget**.

More concretely:

- decode is one token per sequence
- active sequences compete for cache blocks
- if memory runs out, some sequences are temporarily kicked out
- the survivors make forward progress this step

## Short version

`43-58` means:

1. take running sequences one by one
2. see if each can grow by one token
3. if not, preempt other running sequences to free memory
4. if it becomes possible, include the sequence in this decode batch
5. return the batch as `scheduled_seqs, False`

If you want, I can next explain the most confusing line here, `self.running.extendleft(reversed(scheduled_seqs))`, with a queue example.

---

**User**

explain. and it seems at this line  self.running is alwasy empty now, right? @nanovllm/engine/scheduler.py:57 

---

**Cursor**

`self.running.extendleft(reversed(scheduled_seqs))` is putting the just-scheduled decode sequences **back into the `running` queue**, at the front, while **preserving their original left-to-right order**.

## Why put them back?
Earlier in the decode loop, each sequence was removed with:

```python
seq = self.running.popleft()
```

That removal is only temporary so the scheduler can inspect it and decide whether it can decode one more token.

If the sequence is successfully scheduled, it is still an active request after this step, so it must go back into `self.running` for future decode steps.

## Why `extendleft(reversed(...))`?
Because `deque.extendleft()` inserts elements one by one on the **left**, which reverses their order unless you reverse them first.

Example:

```python
scheduled_seqs = [A, B]
```

If you do:

```python
self.running.extendleft(scheduled_seqs)
```

you get:

```python
[B, A, ...]
```

which is backwards.

So the code does:

```python
self.running.extendleft(reversed(scheduled_seqs))
```

which inserts:

- first `B`
- then `A`

and the final deque becomes:

```python
[A, B, ...]
```

So this is a standard trick to preserve order when pushing multiple items to the left side of a deque.

## Is `self.running` always empty at this line?
No, **not always**.

It depends on why the decode loop stopped.

The loop is:

```python
while self.running and num_seqs < self.max_num_seqs:
```

So it stops for one of two reasons:

### Case 1. `self.running` became empty
Then yes, at line 57 it is empty before `extendleft(...)`.

This happens if the scheduler popped and processed all running sequences.

### Case 2. `num_seqs` reached `self.max_num_seqs`
Then `self.running` may still contain leftover sequences that were not scheduled in this step.

So at line 57, `self.running` can still have elements.

## Concrete example

Suppose:

- `self.running = deque([A, B, C, D])`
- `self.max_num_seqs = 2`

Decode loop:

1. pop `A`, schedule it
2. pop `B`, schedule it
3. now `num_seqs == 2`, loop stops

At this moment:

- `scheduled_seqs = [A, B]`
- `self.running = deque([C, D])`

Then line 57 runs:

```python
self.running.extendleft(reversed([A, B]))
```

Final result:

```python
self.running = deque([A, B, C, D])
```

So the scheduled sequences are restored to the front, and the unscheduled ones remain behind them.

## Why this makes sense
The scheduler wants the sequences that just made progress to remain active in `running`, not disappear. And if there were other active sequences that were not scheduled this round, they should still remain in `running` too.

So line 57 is basically:

**“Put the successfully scheduled active sequences back at the front of the active queue.”**

## Short answer
- `extendleft(reversed(scheduled_seqs))` restores scheduled decode sequences to `self.running` while preserving order.
- `self.running` is **not always empty** at that line.
- It is empty only if the loop stopped because all running sequences were consumed.
- If the loop stopped because it hit `max_num_seqs`, `self.running` can still contain unscheduled active sequences.

If you want, I can walk through the whole decode block with a tiny deque example like `[A, B, C]` and show exactly how `running`, `waiting`, and `scheduled_seqs` change at each line.

---

**User**

@nanovllm/engine/scheduler.py:43-58 @nanovllm/engine/llm_engine.py:48-51 after this decode part in schedule(), will the step() function run the whole decoding process for these seqs like the prefill? that is does self.model_runner.call("run", seqs, is_prefill=False) till all seqs performs the decode process?

---

**Cursor**

I’m checking the exact `generate()` and `model_runner.run()` flow so I can answer precisely whether decode happens in one shot or one token at a time.

No. `step()` does **not** run decode until those sequences are fully finished.

It runs **one decode iteration** for the scheduled batch, then returns.

The control flow is:

```48:52:nanovllm/engine/llm_engine.py
def step(self):
    import pdb; pdb.set_trace()
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
```

and the outer loop is:

```74:76:nanovllm/engine/llm_engine.py
while not self.is_finished():
    t = perf_counter()
    output, num_tokens = self.step()
```

So the real behavior is:

1. `schedule()` picks a batch.
2. `model_runner.run(seqs, is_prefill=False)` runs **one batched decode forward pass**.
3. That produces **one next token per scheduled sequence**.
4. `postprocess()` appends that one token and marks any finished sequences.
5. `step()` returns.
6. Later, the outer `generate()` loop calls `step()` again.

## Why it is only one token per sequence in decode

Because `prepare_decode()` only sends the **last token** of each sequence into the model:

```164:180:nanovllm/engine/model_runner.py
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
    ...
    return input_ids, positions
```

Notice:

- `input_ids.append(seq.last_token)`

That means decode input is just **one token per sequence**, not the whole sequence.

Then `run()` samples once:

```208:214:nanovllm/engine/model_runner.py
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

So for decode:

- one batch in
- one sampled token per sequence out

## Compare with prefill
Prefill is also only **one engine step**, not “run until finished.”

The difference is:

- `prefill`: process the whole prompt for each scheduled sequence, then sample its first generated token
- `decode`: process one last-token input for each scheduled sequence, then sample one more token

So both are **single-step operations** inside `step()`, but:
- prefill handles the prompt
- decode advances generation by one token

## What happens after decode
After `postprocess()`:

- finished sequences are removed
- unfinished sequences remain in `running`
- next time `generate()` loops, `schedule()` may pick them again for another one-token decode step

So decode progresses like this:

- step 1: generate token 1
- step 2: generate token 2
- step 3: generate token 3
- ...

not:

- one `step()` call generates the whole completion

## One-line answer
`self.model_runner.call("run", seqs, is_prefill=False)` runs **exactly one batched decode pass**, producing **one token per scheduled sequence**, and the outer `generate()` loop repeatedly calls `step()` until all sequences are finished.

One extra note: `llm_engine.py` currently contains `import pdb; pdb.set_trace()` in `step()`, so if you execute it locally, it will pause in the debugger before scheduling.

---

**User**

@nanovllm/engine/scheduler.py:43-58 then there seems to be a problem. if not self.block_manager.can_append(seq) is triggered and self.preempt(self.running.pop()) is also triggered, there will be one thing in self.waiting. as we mentioned, scheduler.py prioritize prefill over decode. it will prefill this seq in self.waiting again. once prefill is finished, the decode comes again. if the decode process continues, the same problem where not self.block_manager.can_append(seq) will happen again.t

---

**Cursor**

I’m checking `BlockManager.can_append()`, `allocate()`, and `deallocate()` together, because whether this becomes a real loop depends on exactly when a decode step truly needs a new block.

Your concern is reasonable, but it is **not necessarily an infinite loop**. Two details make the behavior less bad than it first looks:

1. `can_append(seq)` only fails at specific block-boundary moments, not on every decode step.
2. A preempted sequence can often be **re-prefilled from prefix cache**, not fully recomputed from scratch.

The key code is:

```93:113:nanovllm/engine/block_manager.py
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]
    if len(seq) % self.block_size == 1:
        assert last_block.hash != -1
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)
    elif len(seq) % self.block_size == 0:
        assert last_block.hash == -1
        token_ids = seq.block(seq.num_blocks-1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
    else:
        assert last_block.hash == -1
```

## First: `can_append()` does not usually fail
This line is subtle:

```python
return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

The right side is a boolean, which acts like:

- `True -> 1`
- `False -> 0`

So this really means:

- if `len(seq) % block_size == 1`, you need **1 free block**
- otherwise, you need **0 free blocks**

So `can_append(seq)` only becomes a problem when the sequence has just crossed into a **new logical block** and needs a fresh physical KV block.

That means this failure does **not** happen every decode step. It only happens at block boundaries.

## Second: what preemption is trying to do
In decode:

```43:58:nanovllm/engine/scheduler.py
# decode
while self.running and num_seqs < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())
        else:
            self.preempt(seq)
            break
    else:
        num_seqs += 1
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))
return scheduled_seqs, False
```

If `seq` cannot grow, the scheduler frees memory by preempting another running sequence. That preempted sequence goes back to `waiting`:

```60:63:nanovllm/engine/scheduler.py
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

So yes, after preemption there will be something in `waiting`, and since scheduling is prefill-first, that request may indeed be re-admitted soon.

## Why that does not automatically cause the same failure immediately
When the preempted sequence is re-admitted, it goes through `allocate()` again:

```59:82:nanovllm/engine/block_manager.py
def allocate(self, seq: Sequence):
    assert not seq.block_table
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        ...
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size
            ...
        ...
        seq.block_table.append(block_id)
```

Important detail: `deallocate()` clears the sequence’s live `block_table`, but it does **not** erase the global hash map of cached full blocks. So when the sequence is re-prefilled:

- full blocks from its old prefix may be found again via `hash_to_block_id`
- those blocks may be reused
- `seq.num_cached_tokens` can increase again

So the preempted request is not always “starting over from zero.” It may get most of its prefix restored cheaply from cache.

## Your suspected loop can happen, but not in the exact way you described
What you’re describing is **thrashing**, and yes, a simplified scheduler like this can thrash under memory pressure.

But the pattern is more like:

- sequence `A` needs a new block
- scheduler preempts `B`
- `A` makes progress
- later `B` is re-admitted, maybe mostly from prefix cache
- later `B` or `A` hits another block boundary
- someone else gets preempted again

So the system can oscillate if the working set is larger than the KV-cache budget.

That is a real tradeoff of this simple design.

## Why it is not an immediate infinite loop
Because after preemption:

- the current sequence now has room to append
- the preempted sequence is no longer in `running`
- on re-prefill, it may reuse cached blocks
- `can_append()` only fails when a new block is needed, not every time

So you do not get:

- preempt
- prefill
- immediate same `can_append` failure
- forever

unless the memory pressure and sequence shapes are especially unlucky.

## The real limitation
The real issue is not “this code is obviously broken,” but rather:

- it is a **very simple recomputation-based preemption policy**
- under heavy load it may cause **repeated prefill/recompute**
- that can hurt throughput or fairness

This is one reason production systems add more careful policies, like:
- chunked prefill
- more nuanced admission control
- better fairness heuristics
- swap/recompute tradeoffs
- decode-priority tuning

## Short answer
Yes, a preempted sequence goes back to `waiting`, and because the scheduler is prefill-first, it may be prefed again soon. But that does **not** automatically mean an infinite loop, because:

- `can_append()` only fails when a new KV block is needed
- re-prefill can reuse prefix-cached blocks
- the preemption freed memory specifically so another sequence could make progress

That said, your intuition about possible **thrashing under tight KV-cache pressure** is valid. This educational scheduler can indeed oscillate more than a production one.

If you want, I can give a concrete toy example with two sequences and a tiny block size, showing exactly when preemption happens and why it does or does not loop.

---

**User**

what is chunked prefill? is it vllm独有? or does sglang also use it?

---

**Cursor**

`chunked prefill` means: **do not process a long prompt in one giant prefill pass**.  
Instead, split the prompt into smaller chunks and run them across multiple scheduler steps.

## What normal prefill does
For a prompt of length 4000, normal prefill does:

- one big forward pass over all 4000 prompt tokens
- then sample the first generated token

This can be good for throughput, but it can also block decode work for other live requests.

## What chunked prefill does
With chunked prefill, the same 4000-token prompt might be processed like:

- step 1: prefill tokens `0..1023`
- step 2: maybe let some decode requests run
- step 3: prefill tokens `1024..2047`
- step 4: maybe decode again
- ...
- final step: after the whole prompt is consumed, sample the next token

So chunked prefill is mainly a **scheduling / latency optimization**.

## Why do this?
Because long prompts can monopolize the GPU.

Chunking helps:
- reduce head-of-line blocking
- improve latency for already-running decode requests
- make the system fairer under mixed workloads
- smooth out memory / compute spikes from huge prompts

The tradeoff is:
- scheduler becomes more complex
- prompt processing is no longer “one request enters, one full prefill happens immediately”

## Is it vLLM 独有?
No. **It is not unique to vLLM.**

It is a general LLM serving idea. vLLM popularized and productized a lot of these scheduling ideas, but chunked prefill itself is not some vLLM-only concept.

## Does SGLang use it?
Yes, **SGLang also uses chunked prefill / similar long-prompt scheduling ideas**.

More generally:
- `vLLM` uses chunked prefill as part of its scheduler/runtime design
- `SGLang` also has mechanisms for handling long prefills efficiently rather than relying only on naive one-shot prompt processing

So the right mental model is:

- **paged KV cache** and **chunked prefill** are modern serving techniques
- they are **not exclusive to vLLM**
- different systems implement them with different schedulers and memory managers

## In one sentence
`chunked prefill` = split long prompt processing into smaller pieces so prefill can be interleaved with other work, and it is **not vLLM-only**; systems like **SGLang also use similar ideas**.

If you want, I can next compare:
1. `normal prefill`
2. `chunked prefill`
3. `decode`  
with a concrete timeline example.

---

**User**

for this repo, what noteworthy features does it implement?

---

**Cursor**

I’m scanning the repo structure and a few core files so I can summarize the notable implemented ideas, not just the engine pieces we already discussed.

I’ve got the high-level claims; now I’m checking the implementation files so I can point to which features are actually present in code.

I’ve confirmed the engine features. I’m checking the parallel linear and sampling layers too, since some of the more interesting pieces live below the scheduler.

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

If you want, I can also rank these features by:
1. hardest to understand,
2. most important for performance, or
3. most “vLLM-like.”

---

**User**

then what are some important features that are important either in vllm or sglang but this repo does not cover?

---

**Cursor**

Yes. This repo covers the **core inference engine ideas**, but it leaves out many features that matter in real `vLLM` or `SGLang` deployments.

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

But it does **not** try to cover the full “production LLM server” surface.

If you want, I can turn this into a two-column comparison:
- `implemented in nano-vllm`
- `common in vLLM/SGLang but missing here`

