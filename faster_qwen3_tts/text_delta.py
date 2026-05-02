"""Helpers for upstream text-delta streaming input."""
from __future__ import annotations

from typing import Callable, List, Sequence


class StreamingTextTokenCommitter:
    """Incrementally commits stable tokenizer ids from raw text deltas."""

    def __init__(self, tokenize_content: Callable[[str], Sequence[int]], token_holdback: int = 1):
        if token_holdback < 0:
            raise ValueError("token_holdback must be >= 0.")
        self._tokenize_content = tokenize_content
        self._token_holdback = int(token_holdback)
        self._text = ""
        self._committed: List[int] = []
        self._finished = False

    @property
    def text(self) -> str:
        return self._text

    @property
    def committed(self) -> List[int]:
        return list(self._committed)

    def push(self, delta: str) -> List[int]:
        if self._finished:
            raise ValueError("Cannot push text after flush().")
        if not isinstance(delta, str):
            raise TypeError(f"text delta must be str, got {type(delta)}.")

        self._text += delta
        current = list(self._tokenize_content(self._text))
        self._ensure_committed_prefix(current)

        commit_to = max(0, len(current) - self._token_holdback)
        if commit_to <= len(self._committed):
            return []

        new_tokens = current[len(self._committed):commit_to]
        self._committed.extend(new_tokens)
        return new_tokens

    def flush(self) -> List[int]:
        if self._finished:
            return []

        current = list(self._tokenize_content(self._text))
        self._ensure_committed_prefix(current)
        new_tokens = current[len(self._committed):]
        self._committed.extend(new_tokens)
        self._finished = True
        return new_tokens

    def _ensure_committed_prefix(self, current: Sequence[int]) -> None:
        prefix = list(current[:len(self._committed)])
        if prefix != self._committed:
            raise ValueError(
                "Tokenizer output changed for already committed text tokens. "
                "Increase token_holdback to keep more unstable suffix tokens buffered."
            )
