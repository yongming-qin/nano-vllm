"""High-level public engine API.

`LLMEngine` is the coordinator that ties together:
- tokenizer I/O
- request creation
- the scheduler
- one local model runner plus optional tensor-parallel worker processes
"""

import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine_annotated.model_runner import ModelRunner
from nanovllm.engine_annotated.scheduler import Scheduler
from nanovllm.engine_annotated.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    """Educational copy of the core generation loop."""

    def __init__(self, model, **kwargs):
        # Only forward config-related kwargs into `Config`; other parameters can
        # still be accepted by higher-level wrappers without breaking here.
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # Rank 0 lives in this process. Additional tensor-parallel ranks are
        # spawned as helper processes and controlled by shared memory/events.
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # EOS is discovered from the tokenizer and then pushed into scheduler
        # termination logic.
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        """Shut down worker processes and distributed resources cleanly."""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Tokenize a prompt if needed and queue it for scheduling."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """Run one engine iteration.

        One iteration is either:
        - a prefill batch for newly admitted prompts, or
        - a decode batch that advances active sequences by one token each
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        # Only finished requests are surfaced to the caller.
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # Positive means prefill throughput; negative encodes decode throughput
        # using the number of advanced sequences, matching the original code.
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """Submit many prompts and wait until all of them finish."""
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # Allow one SamplingParams object to be broadcast to every prompt.
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # Reorder by request id so return order matches submission order.
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
"""High-level public engine API.

`LLMEngine` is the coordinator that ties together:
- tokenizer I/O
- request creation
- the scheduler
- one local model runner plus optional tensor-parallel worker processes
"""

import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine_annotated.model_runner import ModelRunner
from nanovllm.engine_annotated.scheduler import Scheduler
from nanovllm.engine_annotated.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    """Educational copy of the core generation loop."""

    def __init__(self, model, **kwargs):
        # Only forward config-related kwargs into `Config`; other parameters can
        # still be accepted by higher-level wrappers without breaking here.
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # Rank 0 lives in this process. Additional tensor-parallel ranks are
        # spawned as helper processes and controlled by shared memory/events.
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # EOS is discovered from the tokenizer and then pushed into scheduler
        # termination logic.
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        """Shut down worker processes and distributed resources cleanly."""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Tokenize a prompt if needed and queue it for scheduling."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """Run one engine iteration.

        One iteration is either:
        - a prefill batch for newly admitted prompts, or
        - a decode batch that advances active sequences by one token each
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        # Only finished requests are surfaced to the caller.
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # Positive means prefill throughput; negative encodes decode throughput
        # using the number of advanced sequences, matching the original code.
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """Submit many prompts and wait until all of them finish."""
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # Allow one SamplingParams object to be broadcast to every prompt.
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # Reorder by request id so return order matches submission order.
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
