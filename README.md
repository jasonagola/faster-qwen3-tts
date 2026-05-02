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

This benchmark uses one prepared Wimbledon text, slices the first 100, 200, and 500 `o200k_base` tokenizer tokens from it, and replays those token-sized text deltas at 30 tokens/sec. There is no live LLM call in this benchmark. Every downstream TTS number is normalized to the first prepared text token:

```text
first prepared text token == T+0.000s
```

That makes the comparison specific to what a user feels after an LLM starts answering while removing live API variance. The benchmark compares four paths:

- **Vanilla Qwen3-TTS:** full text in, complete audio out.
- **Qwen3-TTS-streaming:** full text in, first back-side streamed audio chunk out.
- **faster-qwen3-tts:** full text in, CUDA graph first streamed audio chunk out.
- **This fork:** text deltas in, CUDA graph first streamed audio chunk out.

The vanilla full-text path uses stock `Qwen3TTSModel.generate_voice_clone(...)`, which returns complete audio rather than yielding a first audio chunk, so the benchmark reports **audio-ready time** for that path. Rows with a TTS `max_new_tokens` cap are excluded; all rows below completed without hitting the cap. Earlier no-sampling runs could hit the cap on the 500-token prefix and produce misleading multi-minute rows, so the README benchmark uses the model's normal sampling path.

Environment: NVIDIA GeForce RTX 5090 32GB, Ubuntu Linux, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, voice clone x-vector mode, prepared Wimbledon text, `o200k_base` tokenizer prefixes, simulated LLM rate 30 tokens/sec, `chunk_size=8`, `token_holdback` values `1` and `8`, `do_sample=True`, `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`, `max_new_tokens=4096`, dtype `bfloat16`, benchmark date `2026-05-02`. Model load and warmup are excluded.

🟩 means at least 1s saved, 🟨 means 0.25-1s saved, and ⬜ means under 0.25s saved.

### Normalized Timeline

`T+0.000s` is the first prepared stream token. Columns are ordered from fastest expected first audio to slowest baseline.

| Target | This repo: Faster + text-delta hb=1 first audio | This repo: Faster + text-delta hb=8 first audio | LLM done | Faster full-text first audio | Qwen3-TTS-streaming first audio | Vanilla Qwen full-text audio ready |
|---:|---:|---:|---:|---:|---:|---:|
| 100 tokens | T+0.358s | T+0.576s | T+3.300s | T+3.572s | T+3.754s | T+20.940s |
| 200 tokens | T+0.341s | T+0.575s | T+6.633s | T+6.892s | T+7.081s | T+38.763s |
| 500 tokens | T+0.348s | T+0.575s | T+16.633s | T+16.893s | T+17.065s | T+101.676s |

### Improvement Breakdown

| Target | Backend audio streaming: Qwen streaming vs vanilla | CUDA graph gain: Faster vs Qwen streaming | Front-side gain hb=1 vs Faster full-text | Front-side gain hb=8 vs Faster full-text | Total gain hb=1 vs vanilla | Total gain hb=8 vs vanilla |
|---:|---:|---:|---:|---:|---:|---:|
| 100 tokens | 🟩 17.186s | ⬜ 0.182s | 🟩 3.214s | 🟩 2.997s | 🟩 20.581s (58.49x) | 🟩 20.364s (36.35x) |
| 200 tokens | 🟩 31.683s | ⬜ 0.189s | 🟩 6.551s | 🟩 6.317s | 🟩 38.422s (113.67x) | 🟩 38.188s (67.41x) |
| 500 tokens | 🟩 84.610s | ⬜ 0.172s | 🟩 16.545s | 🟩 16.318s | 🟩 101.327s (292.17x) | 🟩 101.101s (176.83x) |

In this run, this fork's text-delta first audio stayed around `T+0.34-0.36s` with `token_holdback=1` and around `T+0.58s` with `token_holdback=8`. Qwen3-TTS-streaming removes the wait for complete audio by exposing back-side audio chunks. faster-qwen3-tts then trims the full-text streaming path with CUDA graph capture. This fork removes the remaining front-side wait for the full LLM response.

## Reproduce The Benchmark

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git ../Qwen3-TTS-vanilla
git clone https://github.com/rekuenkdr/Qwen3-TTS-streaming.git ../Qwen3-TTS-streaming
python -m pip install tiktoken

python benchmarks/text_delta_normalized_benchmark.py \
  --targets 100 200 500 \
  --simulated-tokens-per-second 30 \
  --delta-tokenizer openai-o200k \
  --chunk-size 8 \
  --token-holdback 1 \
  --text-delta-token-holdbacks 1 8 \
  --max-new-tokens 4096 \
  --do-sample \
  --vanilla-repo ../Qwen3-TTS-vanilla \
  --streaming-repo ../Qwen3-TTS-streaming
```

The script writes normalized CSV summaries, prepared text recordings, and a README-ready Markdown table under the ignored `text_delta_normalized_benchmark/` directory. The `openai-o200k` tokenizer option requires `tiktoken`; use `--delta-tokenizer qwen` to slice the same source text with the Qwen tokenizer instead. Add `--write-wavs` if you also want generated WAVs for each measured path.

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
