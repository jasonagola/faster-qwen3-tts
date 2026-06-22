# Faster Qwen3-TTS

Real-time Qwen3-TTS inference using CUDA graph capture. No Flash Attention, no vLLM, no Triton. Just `torch.cuda.CUDAGraph`. Supports both streaming and non-streaming generation.

## Install

Requires: Python 3.10+, PyTorch 2.5.1+, NVIDIA GPU with CUDA.

```bash
pip install faster-qwen3-tts
```

**PyTorch compatibility note:** CUDA-graph capture in the fast path is not reliable on `torch<=2.5.0` for this project (capture can fail with "operation not permitted when stream is capturing"). We validated `2.5.1+` as working and set that as the minimum supported version.

**Blackwell note:** RTX 50xx / Blackwell GPUs need CUDA 12.8 PyTorch wheels. If the default setup fails on those cards, install a `cu128` PyTorch build (PyTorch 2.7+).

## Quick Start

### Python

```python
from examples.audio import StreamPlayer  # helper from this repo's examples/
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
ref_audio = "ref_audio.wav"
ref_text = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up "
    "reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach "
    "of training on verifiable outcomes is doomed."
)

# Streaming — yields audio chunks during generation
play = StreamPlayer()
try:
    for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
        text="What do you mean that I'm not real?", language="English",
        ref_audio=ref_audio, ref_text=ref_text,
        chunk_size=8,  # 8 steps ≈ 667ms of audio per chunk
    ):
        play(audio_chunk, sr)
finally:
    play.close()

# Non-streaming — returns all audio at once
audio_list, sr = model.generate_voice_clone(
    text="Hello world!", language="English",
    ref_audio=ref_audio, ref_text=ref_text,
)
```

For local speaker playback from a repo checkout with the example helper:

```bash
pip install sounddevice
```

`examples/audio.py` contains a small `StreamPlayer` helper used by [`examples/streaming_playback.py`](examples/streaming_playback.py). It keeps one output stream open and queues chunks into it. A one-shot player such as `sounddevice.play(audio_chunk, sr)` restarts playback per chunk and can introduce gaps.

### CLI

Voice cloning (reference audio):

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "What do you mean that I'm not real?" \
  --language English \
  --ref-audio ref_audio.wav \
  --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
  --output out.wav
```

CustomVoice (predefined speaker IDs):

```bash
faster-qwen3-tts custom --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --list-speakers
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "What do you mean that I'm not real?" \
  --language English \
  --output out.wav
```

VoiceDesign (instruction-based):

```bash
faster-qwen3-tts design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "Warm, confident narrator with slight British accent" \
  --text "Welcome to the show." \
  --language English \
  --output out.wav
```

Streaming generation to a final WAV file (prints RTF after write):

```bash
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "What do you mean that I'm not real?" \
  --language English \
  --output out.wav \
  --streaming
```

Server mode (keep model hot, stop with `exit`):

```bash
faster-qwen3-tts serve \
  --mode custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --language English \
  --streaming
```

### Demo UI

A minimal web UI that streams audio in real time and shows TTFA and RTF live:

```bash
pip install -e ".[demo]"
python demo/server.py
# open http://localhost:7860
```

Features: voice clone (upload any WAV or use your microphone), voice design (1.7B-VoiceDesign model), streaming/non-streaming toggle, adjustable chunk size, live TTFA/RTF metrics, WAV download.

### OpenAI-compatible API server

`examples/openai_server.py` exposes a `POST /v1/audio/speech` endpoint that follows the OpenAI TTS API contract, so it works out of the box with OpenWebUI, llama-swap, and any other OpenAI-compatible client.

```bash
pip install "faster-qwen3-tts[demo]"
python examples/openai_server.py \
    --ref-audio ref_audio.wav \
    --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
    --language English --port 8000
```

```bash
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model": "tts-1", "input": "Hello world.", "voice": "alloy", "response_format": "wav"}' \
    --output speech.wav
```

To expose multiple voices, pass a JSON file mapping names to reference audio configs — each `voice` value in a request will be routed to the matching entry (`--voices voices.json`). WAV and PCM formats stream chunks as they are generated; MP3 requires `pydub`.

The same server also exposes a WebSocket endpoint for LLM text-delta input streaming:

```text
ws://localhost:8000/v1/audio/speech/deltas
```

The client sends a JSON `start` message, then one or more text `delta` messages, then `done`. The server streams raw PCM binary messages back as soon as Qwen3-TTS can produce audio chunks, followed by a final JSON `{"type":"done"}` message. This endpoint is intended for voice agents that already receive token deltas from an LLM and should not wait for the full response before TTS starts.

```json
{"type":"start","voice":"alloy","response_format":"pcm","language":"English"}
{"type":"delta","text":"The first part of the answer "}
{"type":"delta","text":"arrived from the LLM."}
{"type":"done"}
```

Tunable environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `QWEN_TTS_TEXT_DELTA_IDLE_FILL_INTERVAL_SECONDS` | `0.08` | How often the server yields idle filler while waiting for more text after generation has started. |
| `QWEN_TTS_TEXT_DELTA_IDLE_FILL_MAX_SECONDS` | `8.0` | Maximum idle filler duration before waiting silently for more text. |
| `QWEN_TTS_TEXT_DELTA_IDLE_FILL_TEXT` | space | Text fed during brief upstream LLM pauses so generation can keep moving. |
| `QWEN_TTS_TEXT_DELTA_BASE_XVEC_ONLY` | `1` | Use the lower-latency x-vector voice clone path for Base models by default. |

## Results

Benchmarks include tokenization + inference (apples-to-apples with baseline). RTF > 1.0 = faster than real-time. TTFA measured as time to first playable audio chunk using streaming (chunk_size=8).

### 0.6B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.179 | 3,641ms | 1.307 | 597ms | 7.3x / 6.1x |
| DGX Spark (GB10) | 1.17 | 567ms | 2.56 | 280ms | 2.2x / 2.0x |
| RTX 4090 | 0.82 | 800ms | **4.78** | **156ms** | 5.8x / 5.1x |
| RTX 4060 (Windows) | 0.23 | 2,697ms | **2.26** | **413ms** | 9.8x / 6.5x |
| H100 80GB HBM3 | 0.435 | 1,474ms | **3.884** | **228ms** | 8.9x / 6.5x |

### 1.7B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.183 | 3,573ms | 1.089 | 693ms | 6.0x / 5.2x |
| DGX Spark (GB10) | 1.01 | 661ms | 1.87 | 400ms | 1.9x / 1.7x |
| RTX 4090 | 0.82 | 850ms | **4.22** | **174ms** | 5.1x / 4.9x |
| RTX 4060 (Windows) | 0.23 | 2,905ms | **1.83** | **460ms** | 7.9x / 6.3x |
| H100 80GB HBM3 | 0.439 | 1,525ms | **3.304** | **241ms** | 7.5x / 6.3x |

**Note:** Baseline TTFA values are **streaming TTFA** from the community `Qwen3-TTS-streaming` fork (which adds streaming) or from our **dynamic-cache parity streaming** path (no CUDA graphs) where available. The official `Qwen3-TTS` repo does **not** currently support streaming, so without a streaming baseline TTFA would be **time-to-full-audio**. CUDA graphs uses `generate_voice_clone_streaming(chunk_size=8)` for TTFA. Both include text tokenization for fair comparison. Speedup shows throughput / TTFA improvement. The streaming fork reports additional speedups that appear tied to `torch.compile`; we couldn’t reproduce those on Jetson-class devices where `torch.compile` isn’t available.

**GPU architecture notes:** RTX 4090 (2.5 GHz clocks) outperforms H100 (1.8 GHz) for single-stream workloads. H100's lower baseline (RTF 0.59 vs 4090's 0.82) reflects design optimization for batch processing rather than single-stream inference.

### Benchmark your hardware

Benchmarks run from source. You only need [uv](https://docs.astral.sh/uv/) and `./setup.sh`:

**Linux / macOS / WSL:**

```bash
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
./setup.sh
./benchmark.sh # or ./benchmark.sh 0.6B or ./benchmark.sh 1.7B for a single model
```

**Windows (Native):**

```cmd
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
setup_windows.bat
benchmark_windows.bat   # or benchmark_windows.bat 0.6B / 1.7B / both
```

Results are saved as `bench_results_<GPU_NAME>.json` and audio samples as `sample_0.6B.wav` / `sample_1.7B.wav`.

## Streaming

CUDA graphs support streaming output — audio chunks are yielded during generation with the same per-step performance as non-streaming mode.

### Chunk size vs performance (Jetson AGX Orin, 0.6B)

| chunk_size | TTFA | RTF | Audio per chunk |
|---|---|---|---|
| 1 | 240ms | 0.750 | 83ms |
| 2 | 266ms | 1.042 | 167ms |
| 4 | 362ms | 1.251 | 333ms |
| 8 | 556ms | 1.384 | 667ms |
| 12 | 753ms | 1.449 | 1000ms |
| Non-streaming | — | 1.57 | all at once |

Smaller chunks = lower latency but more decode overhead. `chunk_size=2` is the smallest that stays real-time on Jetson.

**Model seed:** All the different model modes are effectively the same speed. The first time you clone a voice, it takes longer, but later it's cached. Use `benchmarks/compare_modes.py` to reproduce. Example on 0.6B, `chunk_size=8`:

| Mode | TTFA (ms) | RTF | ms/step |
| ---- | --------- | --- | ------- |
| VoiceClone xvec | 152 ± 11 | 5.470 ± 0.032 | 15.2 ± 0.1 |
| VoiceClone full ICL | 149 ± 1 | 5.497 ± 0.026 | 15.2 ± 0.1 |
| CustomVoice | 148 ± 1 | 5.537 ± 0.020 | 15.0 ± 0.1 |

### How streaming works

The CUDA graphs are unchanged — both predictor and talker graphs are replayed per step. The streaming generator yields codec ID chunks every `chunk_size` steps, and the model wrapper decodes each chunk to audio using a sliding window with 25-frame left context (matching the upstream codec's `chunked_decode` pattern) to avoid boundary artifacts.

The Python streaming methods are pull-based generators: they prepare the next chunk when the caller requests it. For realtime local playback, use a queue-backed player such as `StreamPlayer`; blocking after each yielded chunk prevents generation and playback from overlapping.

### Text-delta input streaming

The `generate_*_streaming` methods stream audio out of TTS after the full text prompt is already known. The `stream_*_from_text_deltas` methods stream text into TTS while an upstream LLM is still producing the prompt. This is the front-side streaming path: it is designed to reduce time from "LLM starts answering" to "first playable TTS audio".

```python
text_deltas = ["Hello", ", this is ", "streaming input."]

for audio_chunk, sr, timing in model.stream_custom_voice_from_text_deltas(
    text_deltas=text_deltas,
    speaker="aiden",
    language="English",
    chunk_size=8,
):
    play(audio_chunk, sr)
```

Available methods:

- `stream_custom_voice_from_text_deltas(...)`
- `stream_voice_design_from_text_deltas(...)`
- `stream_voice_clone_from_text_deltas(...)`

The input committer retokenizes the accumulated text with the same assistant wrapper used by normal generation, commits stable content tokens, and holds back the final token by default (`token_holdback=1`) to avoid unstable BPE boundaries. When the input iterator ends, the remaining text tokens and the TTS EOS token are flushed.

To compare full-text output streaming against text-delta input streaming:

```bash
python benchmarks/compare_text_delta_input.py \
  --engines faster \
  --modes custom_voice voice_clone_xvec voice_clone_icl voice_design \
  --multipliers 1 2 3 4 \
  --llm-tokens-per-second 28 \
  --tokens-per-delta 4
```

To run a live OpenAI-to-TTS latency test, set `OPENAI_API_KEY` and run:

```bash
python benchmarks/openai_text_delta_latency.py \
  --openai-model gpt-5.4-mini \
  --engines upstream faster \
  --modes custom_voice \
  --output-limits 100 200 500 1000 \
  --max-seq-len 8192 \
  --max-new-tokens 4096 \
  --chunk-size 8 \
  --write-wavs
```

This warms the TTS path first, then opens a fresh streamed OpenAI request for each engine/mode/token limit and feeds the deltas directly into the TTS generator as they arrive. The same completed text is then used for the full-text baseline, whose timeline starts only after the OpenAI stream finishes. Add `--no-tts-warmup` if you want first-use model setup and CUDA graph capture included in the request timing.

### Normalized front-side streaming benchmark

The normalized benchmark replays prepared LLM-style token deltas at a fixed rate and measures every TTS path from the first prepared text token:

```text
first prepared text token == T+0.000s
```

This removes live API variance and isolates the user-visible question: after the LLM starts answering, how long until audio is playable?

Environment: NVIDIA GeForce RTX 5090 32GB, Ubuntu Linux, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, voice clone x-vector mode, prepared Wimbledon text, `o200k_base` tokenizer prefixes, simulated LLM rate 30 tokens/sec, `chunk_size=8`, `token_holdback` values `1` and `8`, `do_sample=True`, `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`, `max_new_tokens=4096`, dtype `bfloat16`, benchmark date `2026-05-02`. Model load and warmup are excluded.

| Target | Text-delta hb=1 first audio | Text-delta hb=8 first audio | LLM done | Faster full-text first audio | Qwen3-TTS-streaming first audio | Vanilla Qwen full-text audio ready |
|---:|---:|---:|---:|---:|---:|---:|
| 100 tokens | T+0.358s | T+0.576s | T+3.300s | T+3.572s | T+3.754s | T+20.940s |
| 200 tokens | T+0.341s | T+0.575s | T+6.633s | T+6.892s | T+7.081s | T+38.763s |
| 500 tokens | T+0.348s | T+0.575s | T+16.633s | T+16.893s | T+17.065s | T+101.676s |

In that run, text-delta first audio stayed around `T+0.34-0.36s` with `token_holdback=1` and around `T+0.58s` with `token_holdback=8`. Back-side audio streaming removes the wait for complete audio, CUDA graphs reduce full-text streaming overhead, and text-delta input streaming removes the front-side wait for the full LLM response.

Reproduce:

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

The benchmark writes normalized CSV summaries, prepared text recordings, and README-ready Markdown tables under `text_delta_normalized_benchmark/`. Add `--write-wavs` to keep generated WAVs.

### Text-delta validation

Lightweight checks:

```bash
python3 -m py_compile \
  faster_qwen3_tts/model.py \
  faster_qwen3_tts/streaming.py \
  faster_qwen3_tts/text_delta.py \
  examples/openai_server.py \
  benchmarks/compare_text_delta_input.py \
  benchmarks/openai_text_delta_latency.py \
  benchmarks/text_delta_normalized_benchmark.py \
  benchmarks/text_delta_readme_benchmark.py

python3 -m pytest \
  tests/test_text_delta_helpers.py \
  tests/test_text_delta_samples.py \
  tests/test_voice_clone_prompt_api.py \
  tests/test_sampling.py \
  tests/test_sample_rate.py \
  -q
```

Server smoke:

```bash
python examples/openai_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --device cuda \
  --ref-audio ref_audio.wav \
  --ref-text "Reference transcription" \
  --language English

curl http://localhost:8000/health
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"HTTP streaming smoke.","voice":"alloy","response_format":"pcm"}' \
  --output /tmp/qwen3-tts-smoke.pcm
```

For the WebSocket smoke, send the `start`/`delta`/`done` protocol above and assert that at least one binary PCM message and a final JSON `done` are received. MRKS uses this path in `services/voice-worker/app/tts_client.py` through `stream_pcm_from_deltas(...)`.

## Voice Cloning Quality

### Cloning modes

`generate_voice_clone` exposes two modes via `xvec_only`:

| Mode | `xvec_only` | Notes |
|---|---|---|
| Simple (x-vector) | `True` | Speaker embedding only — shorter prefill, clean language switching, no `ref_text` needed |
| Advanced (ICL) | `False` (default) | Full reference audio in context — requires accurate `ref_text`, may produce a brief artifact at the start since it literally continues the sentence `ref_wav` you use |

The default now matches upstream Qwen3-TTS: ICL mode with the reference audio in context. X-vector-only mode remains available as an opt-in for cleaner language switching and shorter prefills.

### Decoder context (ICL mode)

The 12 Hz codec uses a causal `chunked_decode`: each frame is reconstructed using prior frames as acoustic context. In ICL mode the reference audio codec tokens are prepended to the generated tokens before decoding, then the reference portion is trimmed from the output. Without this, the codec decoder starts cold with no voice context — the model generates the right tokens but they get reconstructed in the wrong voice. This is handled automatically.

### Text input streaming vs Non-streaming quality

The original Qwen3TTS implementation supports two mode of generation. It either takes the full input text and prepares the utterance, or it feeds the text progressively. This is the `non_streaming_mode` parameter in the generation methods. The name is maintained from the Qwen3TTS implementation, but I understand it might bring some headaches since here we also have general audio output streaming.
The public API uses `non_streaming_mode=None` as a sentinel, which preserves each method's upstream default unless you override it explicitly.
`generate_voice_clone` and `generate_voice_clone_streaming` resolve `None` to `False`, matching upstream step-by-step text feeding during decode.
`generate_custom_voice`, `generate_custom_voice_streaming`, `generate_voice_design`, and `generate_voice_design_streaming` resolve `None` to `True`, matching the upstream CustomVoice and VoiceDesign defaults.

**Performance impact (RTX 4090, 1.7B, ICL, chunk_size=8):** TTFA is unchanged (≈159ms ± 1ms), and RTF is effectively the same (nsm=False: 4.87 ± 0.01, nsm=True: 4.85 ± 0.01).

### Base-model instruct

`instruct` is available on Base voice cloning, but treat it as experimental when used with `xvec_only=True`. In local testing and upstream-core probing, instruction-following behaved much more predictably in ICL mode (`xvec_only=False`) than in x-vector-only mode.

### ICL Phoneme Artifact

In ICL mode the model's prefill ends with the last codec token of the reference audio, so the first generated token is conditioned on whatever phoneme the reference ends on. If the reference ends mid-word, that phoneme bleeds into the generated speech.

**The fix is applied by default.** The wrapper appends 0.5 s of silence to the reference audio before encoding it, giving the model a clean starting point regardless of how the recording ends. Set `append_silence=False` to match the upstream behavior exactly.

## Quality Samples

### Quality Comparison: Qwen3TTS vs FasterQwen3TTS

We provide side‑by‑side audio samples to compare **Qwen3TTS** (dynamic cache) against **FasterQwen3TTS** (static cache) for both CustomVoice and ICL/voice‑clone. The algorithms are equivalent, but the kernels and reduction order differ, so results are not bit‑identical; the samples let you judge the perceptual impact directly. All samples use the **1.7B** models and cap generation at ~14 seconds so the model can finish naturally.

- `samples/parity/README.md` describes the prompts and model details
- `samples/parity/*.wav` contain 2 voices × 2 prompts × {static,dynamic}

**CustomVoice (aiden) – Prompt 1**

<audio controls src="samples/parity/custom_aiden_gen1_static.wav"></audio>
<audio controls src="samples/parity/custom_aiden_gen1_dynamic.wav"></audio>

**CustomVoice (aiden) – Prompt 2**

<audio controls src="samples/parity/custom_aiden_gen2_static.wav"></audio>
<audio controls src="samples/parity/custom_aiden_gen2_dynamic.wav"></audio>

**CustomVoice (serena) – Prompt 1**

<audio controls src="samples/parity/custom_serena_gen1_static.wav"></audio>
<audio controls src="samples/parity/custom_serena_gen1_dynamic.wav"></audio>

**CustomVoice (serena) – Prompt 2**

<audio controls src="samples/parity/custom_serena_gen2_static.wav"></audio>
<audio controls src="samples/parity/custom_serena_gen2_dynamic.wav"></audio>

**ICL (ref_audio.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_gen1_dynamic.wav"></audio>

**ICL (ref_audio.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_gen2_dynamic.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_2_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_2_gen1_dynamic.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_2_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_2_gen2_dynamic.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_3_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_3_gen1_dynamic.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_3_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_3_gen2_dynamic.wav"></audio>

### non_streaming_mode Comparison (ICL)

We provide side‑by‑side samples comparing **non_streaming_mode=False** vs **True** for ICL voice cloning.
All samples use the **1.7B** model with `xvec_only=False`.

- `samples/non_streaming_mode/README.md` describes prompts, settings, and filenames
- `samples/non_streaming_mode/*.wav` contain 3 references × 2 prompts × {nsm_false,nsm_true}

**ICL (ref_audio.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen1_nsm_true.wav"></audio>

**ICL (ref_audio.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen2_nsm_true.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen1_nsm_true.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen2_nsm_true.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen1_nsm_true.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen2_nsm_true.wav"></audio>

## Parity

We maintain parity with upstream Qwen3‑TTS in two layers, and document where (and why) the fast path can differ numerically. When we say **Qwen3TTS vs FasterQwen3TTS**, we are comparing the upstream dynamic‑cache path against our static‑cache CUDA‑graph path.

- **Fast path (static cache + CUDA graphs):** Streaming and non‑streaming share the same decode core and match upstream for the initial window where artifacts are most audible. Tests enforce this prefix parity deterministically.
- **Parity mode (dynamic cache, tests only):** A dynamic‑cache decode path (no CUDA graphs) that calls `talker.generate(...)` is used in tests to prove exact token‑level equality against upstream for all model types.

**Why can static cache differ from dynamic cache?** The math is equivalent, but the kernel path is not. Static cache uses a fixed max‑length KV buffer and an explicit attention mask, which often selects a different SDPA kernel than the dynamic cache path (shorter K/V, `is_causal=True`, mask‑free). In BF16/TF32, different kernel/reduction orders are not bit‑exact, so the outputs can differ slightly even when the algorithm is the same.

**Parity streaming note:** The dynamic‑cache parity streaming path is intentionally slow. On an RTX 4090 it measured ~0.77s TTFA (chunk_size=8) and ~1.17s TTFA (chunk_size=12), versus ~0.16–0.18s TTFA in the fast CUDA‑graph path. Use parity streaming only for validation, not performance.

Tests live in `tests/test_e2e_parity.py` and cover:

- Voice clone (x‑vector) prefix parity vs upstream
- Streaming vs non‑streaming parity (fast path)
- CustomVoice full equality (parity mode)
- VoiceDesign full equality (parity mode)
- Voice clone ICL full equality (parity mode)

You can control the model IDs used by tests via environment variables:

```
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
QWEN_TTS_CUSTOM_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_TTS_VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```
## How It Works

Qwen3-TTS runs two autoregressive transformers per decode step:
1. **Talker** (28 layers): generates the first codebook token from text
2. **Code Predictor** (5 layers): generates 15 additional codebook tokens

A single step involves ~500 small CUDA kernel launches with Python overhead between them. The GPU spends more time waiting for the next kernel than computing.

CUDA graphs capture the entire decode step and replay it as a single GPU operation:

1. **Static KV cache**: pre-allocated fixed-size tensors (no dynamic allocation)
2. **Model's own forward**: SDPA + RoPE via the model's native attention layers
3. **Graph capture**: `torch.cuda.CUDAGraph` for both predictor and talker
4. **Padded attention**: attention mask handles variable-length KV within fixed buffers

### Per-component breakdown (Jetson AGX Orin, 0.6B)

| Component | Before | After |
|---|---|---|
| Talker (28 layers) | 75ms | 12ms |
| Predictor (15 steps) | 190ms | 26ms |
| Overhead | 65ms | 16ms |
| **Total per step** | **330ms** | **54ms** |

## Voice Cloning with Precomputed Speaker Embeddings

For production use, extract the speaker embedding once and reuse it:

```bash
# 1. Extract speaker embedding from reference audio (one-time, ~10s)
python examples/extract_speaker.py --ref_audio voice.wav --output speaker.pt

# 2. Generate speech with CUDA graphs (real-time)
python examples/generate_with_embedding.py --speaker speaker.pt --text "Hello!" --language English --output en.wav
python examples/generate_with_embedding.py --speaker speaker.pt --text "Bonjour!" --language French --output fr.wav
python examples/generate_with_embedding.py --speaker speaker.pt --text "Hallo!" --language German --output de.wav
```

The speaker embedding is a 4KB file (2048-dim bf16 vector). In `x_vector_only` mode:
- **No accent bleed**: native pronunciation per language
- **Shorter prefill**: 10 tokens vs ~80+ in full ICL clone mode
- **No ref audio at runtime**: just the 4KB embedding file

You can now pass a precomputed prompt directly to the public APIs. The wrapper
accepts either:
- the raw `prompt_items` list returned by `create_voice_clone_prompt(...)`
- or the lower-level dict form produced by `_prompt_items_to_voice_clone_prompt(...)`

```python
import torch
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# 1) Compute prompt_items once from reference audio
prompt_items = model.model.create_voice_clone_prompt(
    ref_audio="voice.wav",
    ref_text="",
    x_vector_only_mode=True,
)

# 2) You can pass prompt_items directly
audio_list, sr = model.generate_voice_clone(
    text="Hello world!",
    language="English",
    voice_clone_prompt=prompt_items,
)

# 3) Or save just the speaker embedding and rebuild the compact dict form
spk_emb = prompt_items[0].ref_spk_embedding

torch.save(spk_emb.detach().cpu(), "speaker.pt")

spk_emb = torch.load("speaker.pt", weights_only=True).to(model.device)

voice_clone_prompt = {
    "ref_spk_embedding": [spk_emb],
}

audio_list, sr = model.generate_voice_clone(
    text="Hello world!",
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
```

When `voice_clone_prompt` is provided, prompt extraction from `ref_audio` is skipped.
For x-vector-only prompts, `ref_text` is ignored.
For ICL precomputed prompts, pass `x_vector_only_mode=[False]`, `icl_mode=[True]`,
and a non-`None` `ref_code`, and keep `ref_text` populated.

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team
- [Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) for ideas and code we adapted for streaming
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) for inspiration on CUDA graph usage
- NVIDIA for providing the Jetson AGX Orin board
