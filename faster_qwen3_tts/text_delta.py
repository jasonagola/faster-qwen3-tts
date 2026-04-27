"""Helpers for upstream text-delta streaming input."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple


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


def split_token_budget_deltas(text: str, tokens_per_delta: int, count_tokens: Callable[[str], int]) -> List[str]:
    """Split text into word-preserving deltas with an approximate token budget."""
    if tokens_per_delta <= 0:
        raise ValueError("tokens_per_delta must be > 0.")
    words = text.split()
    if not words:
        return []

    deltas = []
    current_words = []
    for word in words:
        candidate_words = current_words + [word]
        candidate_text = " ".join(candidate_words)
        if current_words and count_tokens(candidate_text) > tokens_per_delta:
            deltas.append(" ".join(current_words) + " ")
            current_words = [word]
        else:
            current_words = candidate_words

    if current_words:
        deltas.append(" ".join(current_words))
    return deltas


def token_counted_delta_delays(
    deltas: Iterable[str],
    llm_tokens_per_second: float,
    count_tokens: Callable[[str], int],
) -> List[Tuple[str, int, float]]:
    """Attach a simulated LLM delay to each text delta based on token count."""
    if llm_tokens_per_second <= 0:
        raise ValueError("llm_tokens_per_second must be > 0.")

    delayed = []
    for delta in deltas:
        token_count = max(1, count_tokens(delta))
        delayed.append((delta, token_count, token_count / llm_tokens_per_second))
    return delayed
