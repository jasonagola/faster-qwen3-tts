# Faster Qwen3-TTS With Text-Delta Input Streaming

This fork adds **text-delta input streaming** to [`andimarafioti/faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts). The base project already streams audio chunks after TTS starts. This fork lets TTS start while an upstream LLM is still producing text.

The goal is lower time to first audio in LLM-to-TTS systems:

```text
base path:       wait for complete LLM text -> Qwen TTS -> streamed audio chunks
this fork:       LLM text deltas -> stable text tokens -> Qwen TTS -> streamed audio chunks
```

Existing `generate_*` and `generate_*_streaming` APIs are unchanged.

## What This Fork Adds

| Capability | Upstream repo | This fork |
|---|---|---|
| Full-text non-streaming TTS | `generate_* (...)` | Unchanged |
| Full-text input, streamed audio output | `generate_*_streaming(...)` | Unchanged |
| LLM-style text deltas, streamed audio output | Not available | `stream_*_from_text_deltas(...)` |
| Server/WebSocket/OpenAI-compatible API changes | Existing upstream behavior only | Not changed here |
| Latency/stability knob for streamed text | Not applicable | `token_holdback` |

Use `generate_*_streaming(...)` when your application already has the full utterance. Use `stream_*_from_text_deltas(...)` when text is arriving from a streaming LLM and time to first audio matters.

Available text-delta APIs:

- `stream_custom_voice_from_text_deltas(...)`
- `stream_voice_design_from_text_deltas(...)`
- `stream_voice_clone_from_text_deltas(...)`

Each yields the same output shape as the existing streaming APIs:

```python
(audio_chunk, sample_rate, timing)
```

## Quick Example

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device="cuda:0",
)

text_deltas = ["Hello", ", this is ", "streaming input."]

for audio_chunk, sr, timing in model.stream_custom_voice_from_text_deltas(
    text_deltas=text_deltas,
    speaker="Ryan",
    language="English",
    chunk_size=8,
    token_holdback=1,
):
    # Send audio_chunk to your player, WebRTC track, or WAV writer.
    pass
```

For comparison, the existing full-text audio-output streaming path still works exactly as before:

```python
for audio_chunk, sr, timing in model.generate_custom_voice_streaming(
    text="Hello, this is streaming output from a complete text prompt.",
    speaker="Ryan",
    language="English",
    chunk_size=8,
):
    pass
```

The difference is when TTS is allowed to start. `generate_custom_voice_streaming(...)` waits until the full `text` string exists. `stream_custom_voice_from_text_deltas(...)` can begin after stable tokens arrive from the input iterator.

## API Comparison

| API | Input | Output | Notes |
|---|---|---|---|
| `generate_*_streaming(...)` | Complete text string | Qwen TTS audio chunks | Full-text input path. Audio can stream out, but TTS waits until all text is available. |
| `stream_*_from_text_deltas(...)` | Iterable of partial text chunks | Qwen TTS audio chunks | Text-delta input path. TTS can begin while the upstream LLM is still writing. |
| OpenAI [`/v1/audio/speech`](https://platform.openai.com/docs/api-reference/audio/createSpeech) | Complete `input` text | Audio response / streamed audio bytes | Standard speech API pattern: text is complete before TTS starts. |
| OpenAI [Responses streaming](https://platform.openai.com/docs/guides/streaming) | Prompt/messages | Text deltas such as [`response.output_text.delta`](https://platform.openai.com/docs/api-reference/responses-streaming/response/output_text/delta) | LLM text streaming pattern; this fork bridges those deltas into Qwen TTS. |

## Token Holdback

The text-delta committer retokenizes accumulated text with the same assistant wrapper used by normal generation. It commits stable content tokens and keeps a small suffix buffered so BPE boundaries can settle.

| `token_holdback` | Behavior |
|---|---|
| `0` | Lowest latency. Highest risk that an early BPE token later changes when the next characters arrive. |
| `1` | Default balanced mode. Keeps one token of local lookahead before feeding TTS. |
| `3+` | More conservative local lookahead. Higher latency, but can help phrasing when upstream deltas split words or clauses aggressively. |

Full-text generation remains best when maximum prosody and future sentence context matter more than latency.

## Benchmark: Normalized LLM-to-TTS Timeline

This benchmark records a real OpenAI Responses stream once, then replays the exact same text deltas with the same inter-delta timing into each TTS path. Every downstream TTS number is normalized to the first LLM text delta:

```text
first OpenAI response.output_text.delta == T+0.000s
```

That makes the comparison specific to what a user feels after the LLM starts answering. The benchmark separates two different gains:

- **Backend implementation gain:** FasterQwen3TTS vs vanilla `Qwen3TTSModel`.
- **Front-side input streaming gain:** text deltas into TTS vs waiting for full LLM text.

The vanilla full-text path uses `Qwen3TTSModel.generate_custom_voice(...)`, which returns complete audio rather than yielding a first audio chunk, so the benchmark reports **audio-ready time** for that path. Stock vanilla Qwen3-TTS does not provide audio-output streaming, so there is no vanilla "first audio chunk" column in the headline table. Rows with a TTS `max_new_tokens` cap are excluded; all rows below completed without hitting the cap.

Environment: NVIDIA GeForce RTX 5090 32GB, Ubuntu Linux, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, speaker `Ryan`, `gpt-5.4-mini`, `chunk_size=8`, `token_holdback=1`, `do_sample=True`, `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`, `max_new_tokens=4096`, dtype `bfloat16`, benchmark date `2026-05-02`. Model load and warmup are excluded.

🟩 means at least 1s saved, 🟨 means 0.25-1s saved, and ⬜ means under 0.25s saved.

### Normalized Timeline

`T+0.000s` is the first LLM text token.

| Target | This repo: Faster + text-delta first audio | LLM done | Faster + full-text first audio | Vanilla Qwen full-text audio ready |
|---:|---:|---:|---:|---:|
| 100 tokens | T+0.312s | T+1.109s | T+1.369s | T+28.456s |
| 200 tokens | T+0.265s | T+3.036s | T+3.302s | T+43.764s |
| 500 tokens | T+0.263s | T+3.105s | T+3.373s | T+94.484s |

### Improvement Breakdown

| Target | Backend gain: Faster full-text vs vanilla full-text | Front-side gain: Faster text-delta vs Faster full-text | Total gain: this repo vs vanilla full-text |
|---:|---:|---:|---:|
| 100 tokens | 🟩 27.086s | 🟩 1.058s | 🟩 28.144s (91.21x) |
| 200 tokens | 🟩 40.462s | 🟩 3.037s | 🟩 43.499s (165.15x) |
| 500 tokens | 🟩 91.111s | 🟩 3.111s | 🟩 94.222s (359.25x) |

In this run, Faster text-delta first audio stayed around `T+0.26-0.31s` after the first LLM token. The backend implementation removes most of the vanilla full-audio wait, and front-side text streaming removes the remaining wait for the full LLM response.

## Reproduce The Benchmark

```bash
export OPENAI_API_KEY=...

python benchmarks/text_delta_normalized_benchmark.py \
  --openai-model gpt-5.4-mini \
  --targets 100 200 500 \
  --chunk-size 8 \
  --token-holdback 1 \
  --max-new-tokens 4096
```

The script writes normalized CSV summaries, OpenAI text recordings, and a README-ready Markdown table under the ignored `text_delta_normalized_benchmark/` directory. Add `--write-wavs` if you also want generated WAVs for each measured path. A patched vanilla Qwen3-TTS text-delta path can be measured with `--include-vanilla-text-delta`, but it is intentionally omitted from the default table because stock vanilla Qwen3-TTS does not stream audio.

## Text-Delta Samples

Each pair uses the same text and generation settings except for the input path: text-delta input streaming vs complete-text audio-output streaming.

**CustomVoice 200-token sample**

<audio controls src="samples/text_delta_streaming/custom_voice_200_text_delta.wav"></audio>
<audio controls src="samples/text_delta_streaming/custom_voice_200_fulltext.wav"></audio>

**VoiceDesign 100-token sample**

<audio controls src="samples/text_delta_streaming/voice_design_100_text_delta.wav"></audio>
<audio controls src="samples/text_delta_streaming/voice_design_100_fulltext.wav"></audio>

**Voice clone x-vector 100-token sample**

<audio controls src="samples/text_delta_streaming/voice_clone_xvec_100_text_delta.wav"></audio>
<audio controls src="samples/text_delta_streaming/voice_clone_xvec_100_fulltext.wav"></audio>

## Install

Requires Python 3.10+, PyTorch 2.5.1+, and an NVIDIA GPU with CUDA.

```bash
git clone -b feature/text-delta-input-streaming https://github.com/jasonagola/faster-qwen3-tts.git
cd faster-qwen3-tts
pip install -e .
```

RTX 50xx / Blackwell GPUs need CUDA 12.8 PyTorch wheels. If the default PyTorch install fails on those cards, install a `cu128` PyTorch build.

## Brief faster-qwen3-tts Guide

This fork keeps the base faster-qwen3-tts model loading, CLI, and full-text APIs intact. The underlying project uses CUDA graph capture for faster single-stream Qwen3-TTS inference, but you do not need to manage CUDA graphs directly.

Load a model:

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device="cuda:0",
)
```

Generate full-text audio chunks:

```python
for audio_chunk, sr, timing in model.generate_custom_voice_streaming(
    text="This complete sentence is known before TTS starts.",
    speaker="Ryan",
    language="English",
    chunk_size=8,
):
    pass
```

Generate one-shot audio:

```python
audio_list, sr = model.generate_custom_voice(
    text="Hello world.",
    speaker="Ryan",
    language="English",
)
```

Common modes:

| Mode | Full-text streaming | Text-delta input streaming |
|---|---|---|
| CustomVoice | `generate_custom_voice_streaming(...)` | `stream_custom_voice_from_text_deltas(...)` |
| VoiceDesign | `generate_voice_design_streaming(...)` | `stream_voice_design_from_text_deltas(...)` |
| Voice clone | `generate_voice_clone_streaming(...)` | `stream_voice_clone_from_text_deltas(...)` |

CLI usage from the base project remains available for full-text generation:

```bash
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker Ryan \
  --text "A complete text prompt." \
  --language English \
  --output out.wav \
  --streaming
```

For deeper details on the base CUDA graph implementation, model parity, and hardware benchmark matrix, see the upstream project.

## Tests

Lightweight tests:

```bash
python3 -m py_compile \
  faster_qwen3_tts/model.py \
  faster_qwen3_tts/streaming.py \
  faster_qwen3_tts/text_delta.py \
  benchmarks/text_delta_normalized_benchmark.py \
  benchmarks/text_delta_readme_benchmark.py

python3 -m pytest \
  tests/test_text_delta_helpers.py \
  tests/test_text_delta_samples.py \
  tests/test_voice_clone_prompt_api.py \
  tests/test_sampling.py \
  tests/test_sample_rate.py \
  -q

git diff --check
```

Manual GPU validation should check that every `stream_*_from_text_deltas(...)` mode yields nonempty PCM chunks, the sample rate is valid, concatenated WAV writing succeeds, the matching full-text streaming path still produces audio, and benchmark rows are not capped by `max_new_tokens`.

## Scope

This is a Python API addition. Server/WebSocket/OpenAI-compatible protocol changes are intentionally out of scope for this branch. Generated benchmark CSV/JSONL/WAV dumps and `.env` files are ignored; only curated sample WAVs are committed.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Built on [`andimarafioti/faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) and Qwen3-TTS.
