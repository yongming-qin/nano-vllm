"""Annotated engine package for learning how nano-vllm works.

This package mirrors `nanovllm.engine`, but the files contain more teaching
comments and docstrings. The intent is to make the control flow easier to
study without changing the original implementation.
"""

from nanovllm.engine_annotated.llm_engine import LLMEngine
from nanovllm.engine_annotated.sequence import Sequence, SequenceStatus
from nanovllm.engine_annotated.scheduler import Scheduler
from nanovllm.engine_annotated.block_manager import BlockManager, Block
from nanovllm.engine_annotated.model_runner import ModelRunner
