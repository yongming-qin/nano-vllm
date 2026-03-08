"""Sequence state used by the scheduler and model runner.

The original implementation is intentionally compact. This copy adds
explanations for what each field represents during prefill, decode, and
KV-cache management.
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Lifecycle of a request inside the engine."""

    # The request has been admitted into the engine, but has not yet been
    # scheduled onto the GPU.
    WAITING = auto()

    # The request owns KV-cache blocks and participates in decoding rounds.
    RUNNING = auto()

    # Generation stopped because EOS was produced or max_tokens was reached.
    FINISHED = auto()


class Sequence:
    """Mutable per-request state tracked across the whole generation loop.

    A sequence stores:
    - the full token history (`token_ids`)
    - how many tokens belong to the original prompt
    - how many prompt tokens were reused from prefix cache
    - which KV-cache blocks currently back this request

    `Sequence` is also serialized and shipped between worker processes, so the
    custom pickle helpers at the bottom intentionally keep the payload small.
    """

    # KV cache is managed in fixed-size blocks. This class-level default is only
    # used for reasoning about sequence layout; the scheduler/config decide the
    # actual engine-wide block size and that value must stay aligned.
    block_size = 256

    # Monotonic request id generator so outputs can be returned in submission
    # order even if requests finish at different times.
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        # Copy the prompt so later appends do not mutate the caller's list.
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)

        # Number of prompt tokens whose KV entries were already present in the
        # prefix cache and therefore do not need to be recomputed in prefill.
        self.num_cached_tokens = 0

        # Maps logical block index -> physical KV-cache block id.
        self.block_table = []

        # Sampling parameters are copied onto the sequence so scheduling and
        # model execution can work with a single state object.
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """How many tokens were generated after the prompt."""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """How many full cache blocks are reused from prefix cache."""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """How many KV blocks are needed to store the sequence so far."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """Number of real tokens in the final (possibly partial) block."""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """Return the token ids that belong to logical block `i`."""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """Append one sampled token after a decode step."""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """Compact pickle payload used by multi-process tensor parallelism.

        Before any generation happens we need the entire prompt token list.
        Afterwards, workers only need lightweight metadata plus the latest token
        for decode, because the historical KV states already live on device.
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
