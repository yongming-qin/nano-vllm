"""Admission control and scheduling policy for generation requests.

The scheduler decides:
1. which waiting prompts can enter the batch (`prefill`)
2. which running requests advance by one token (`decode`)
3. when to preempt requests if KV-cache space runs out
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine_annotated.block_manager import BlockManager
from nanovllm.engine_annotated.sequence import Sequence, SequenceStatus


class Scheduler:
    """Small scheduler that alternates between prompt prefill and decoding."""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # Requests enter `waiting`, move to `running`, and leave the system once
        # marked `FINISHED`.
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Pick the next work to run.

        Returns `(scheduled_seqs, is_prefill)`.

        Policy:
        - Prefer prefill first because new prompts may admit a large amount of
          parallel work at once.
        - If no waiting request fits, run one decode step for as many active
          sequences as possible.
        """

        # Prefill phase: admit new prompts until sequence count, token budget,
        # or KV-cache capacity says stop.
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)

            # Prefix-cached tokens do not require model compute, so only the
            # uncached suffix contributes to the effective prefill workload.
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode phase: each scheduled sequence produces exactly one next token.
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # If there is no free space to extend the current sequence, preempt
            # another running sequence (or this one if it is the only choice).
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

        # Put scheduled requests back at the front in their original order.
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """Evict a running sequence back to the waiting queue.

        This implementation uses recomputation-style preemption: it simply frees
        the sequence's blocks and later re-prefills the sequence from its token
        history when resources are available again.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """Apply sampled tokens and retire finished requests."""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
