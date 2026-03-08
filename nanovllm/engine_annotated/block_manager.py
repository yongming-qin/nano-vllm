"""Block allocator and prefix-cache bookkeeping for the KV cache.

This file is one of the core ideas behind vLLM-style serving:
the cache is split into fixed-size blocks, and prompt prefixes can be reused by
hashing full blocks of tokens. The scheduler asks this manager whether it can
allocate or extend a sequence before launching model work.
"""

from collections import deque

import numpy as np
import xxhash

from nanovllm.engine_annotated.sequence import Sequence


class Block:
    """Metadata for one physical KV-cache block."""

    def __init__(self, block_id):
        self.block_id = block_id

        # How many live sequences currently point at this block. Prefix-cached
        # blocks can be shared, so this is not always 1.
        self.ref_count = 0

        # Hash of the full token block plus prefix hash. Partial blocks are not
        # hashable/reusable, so they keep `-1`.
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """Record the hash and exact tokens stored in this block."""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """Prepare a recycled block for a fresh allocation."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Owns free/used blocks and prefix-cache lookup tables."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # Maps a prefix-aware block hash to the physical block id that currently
        # contains those tokens.
        self.hash_to_block_id: dict[int, int] = dict()

        # Free ids are kept in FIFO order so allocation is deterministic.
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """Hash one full token block, optionally chained to the prefix hash.

        Chaining the previous block's hash means two identical token blocks only
        collide if their full prefix history is also the same. That lets the
        cache represent an entire prefix tree using block-level hashes.
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Move a block from the free pool to the used pool."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """Return an unreferenced block back to the free pool."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Whether the sequence can reserve blocks for its current prompt."""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Assign physical blocks for a sequence's current token history.

        For prompt admission, we walk logical blocks from left to right. Once a
        cache miss happens, every later block becomes a miss too because the
        prefix hash chain changes.
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)

            # Only full blocks are eligible for prefix caching. The final
            # partial block remains unhashed until it becomes full later.
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            # Re-check token ids even if the hash exists. This guards against a
            # theoretical hash collision.
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size

                # A cached block may already be shared by active sequences. If
                # so, just bump the reference count instead of reallocating it.
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)

            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Release all blocks currently owned or referenced by a sequence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Whether one more decode token can be appended safely.

        A new physical block is only required when the next token would start a
        fresh block, which is why the boolean expression becomes `1` or `0`.
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Update block bookkeeping before/after a decode append.

        There are three cases:
        - `len(seq) % block_size == 1`: the previously appended token started a
          new block, so we need to allocate storage for that block.
        - `len(seq) % block_size == 0`: the current last block just became full,
          so we can hash it and expose it to prefix cache.
        - otherwise: we are still filling a partial trailing block.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # The previous block is already full and cached; decoding just
            # stepped into a brand-new trailing block.
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # The trailing partial block has become a full cacheable block.
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # Still writing into a partial block, so no hash yet.
            assert last_block.hash == -1
