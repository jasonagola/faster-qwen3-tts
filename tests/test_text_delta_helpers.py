import pytest

from faster_qwen3_tts.text_delta import (
    StreamingTextTokenCommitter,
    split_token_budget_deltas,
    token_counted_delta_delays,
)


def test_text_committer_holds_unstable_suffix_and_flushes_final_tokens():
    tokenizations = {
        "": [],
        "Hel": [101],
        "Hello": [10],
        "Hello,": [10, 11],
        "Hello, world": [10, 11, 12],
        "Hello, world.": [10, 11, 12, 13],
    }
    committer = StreamingTextTokenCommitter(tokenizations.__getitem__, token_holdback=1)

    assert committer.push("Hel") == []
    assert committer.push("lo") == []
    assert committer.push(",") == [10]
    assert committer.push(" world") == [11]
    assert committer.push(".") == [12]
    assert committer.flush() == [13]
    assert committer.flush() == []


def test_text_committer_accepts_empty_chunks():
    tokenizations = {
        "": [],
        "go": [1, 2],
    }
    committer = StreamingTextTokenCommitter(tokenizations.__getitem__, token_holdback=0)

    assert committer.push("") == []
    assert committer.push("go") == [1, 2]
    assert committer.flush() == []


def test_text_committer_rejects_committed_prefix_changes():
    def tokenize(text):
        return [1] if text == "a" else [2]

    committer = StreamingTextTokenCommitter(tokenize, token_holdback=0)
    assert committer.push("a") == [1]
    with pytest.raises(ValueError, match="already committed"):
        committer.push("b")


def test_split_token_budget_deltas_preserves_words_and_delays():
    deltas = split_token_budget_deltas(
        "one two three four",
        tokens_per_delta=2,
        count_tokens=lambda text: len(text.split()),
    )

    assert deltas == ["one two ", "three four"]

    delayed = token_counted_delta_delays(
        deltas,
        llm_tokens_per_second=4,
        count_tokens=lambda text: len(text.split()),
    )
    assert delayed == [("one two ", 2, 0.5), ("three four", 2, 0.5)]
