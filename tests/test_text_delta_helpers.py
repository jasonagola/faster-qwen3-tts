import inspect

import pytest

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.text_delta import StreamingTextTokenCommitter


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


def test_text_committer_handles_punctuation_and_whitespace_deltas():
    tokenizations = {
        "": [],
        "Hello": [10],
        "Hello,": [10, 11],
        "Hello, ": [10, 11],
        "Hello, world": [10, 11, 12],
        "Hello, world!": [10, 11, 12, 13],
    }
    committer = StreamingTextTokenCommitter(tokenizations.__getitem__, token_holdback=1)

    assert committer.push("Hello") == []
    assert committer.push(",") == [10]
    assert committer.push(" ") == []
    assert committer.push("world") == [11]
    assert committer.push("!") == [12]
    assert committer.flush() == [13]


def test_text_committer_rejects_committed_prefix_changes():
    def tokenize(text):
        return [1] if text == "a" else [2]

    committer = StreamingTextTokenCommitter(tokenize, token_holdback=0)
    assert committer.push("a") == [1]
    with pytest.raises(ValueError, match="already committed"):
        committer.push("b")


def test_text_delta_streaming_public_methods_exist():
    for method_name in (
        "stream_custom_voice_from_text_deltas",
        "stream_voice_design_from_text_deltas",
        "stream_voice_clone_from_text_deltas",
    ):
        signature = inspect.signature(getattr(FasterQwen3TTS, method_name))
        assert "text_deltas" in signature.parameters
        assert "token_holdback" in signature.parameters
        assert "chunk_size" in signature.parameters
