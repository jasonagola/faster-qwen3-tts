#!/usr/bin/env python3
"""Benchmark LLM-to-TTS timing normalized to the first LLM text token.

The benchmark records a real OpenAI Responses stream once per target length,
then replays the same text deltas with the same inter-delta timing into each
TTS implementation. This keeps the LLM side identical while comparing:

- vanilla Qwen3-TTS full-text input
- FasterQwen3TTS full-text input
- FasterQwen3TTS text-delta input

It can optionally measure a patched vanilla Qwen3-TTS text-delta path when
--include-vanilla-text-delta is passed, but that is not part of the default
public comparison because stock vanilla Qwen3-TTS does not stream audio.

Output is written under --out-dir, which is ignored by git.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import soundfile as sf
import torch


THIS_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_REPO))

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


@dataclass
class TextRecording:
    target_tokens: int
    text: str
    first_token_s: float
    done_s: float
    output_tokens: Optional[int]
    status: Optional[str]
    delta_events: list[dict]
    timeline: list[dict]

    @property
    def done_after_first_token_s(self) -> float:
        return self.done_s - self.first_token_s


@dataclass
class TtsMeasurement:
    label: str
    first_audio_s: Optional[float]
    audio_done_s: Optional[float]
    chunks: int = 0
    samples: int = 0
    sample_rate: Optional[int] = None
    hit_cap: Optional[bool] = None
    available: bool = True
    note: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare vanilla Qwen3-TTS and FasterQwen3TTS normalized to first LLM token."
    )
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-timeout", type=float, default=180.0)
    parser.add_argument("--targets", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--custom-model", default=DEFAULT_CUSTOM_MODEL)
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--language", default="English")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--token-holdback", type=int, default=1)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--do-sample", dest="do_sample", action="store_true")
    sampling.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--out-dir", type=Path, default=Path("text_delta_normalized_benchmark"))
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--include-vanilla-text-delta", action="store_true")
    parser.add_argument("--write-wavs", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def metric(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.3f}"


def tplus(value: Optional[float]) -> str:
    return "" if value is None else f"T+{value:.3f}s"


def improvement(value: Optional[float]) -> str:
    if value is None:
        return ""
    marker = "🟩" if value >= 1.0 else "🟨" if value >= 0.25 else "⬜"
    return f"{marker} {value:.3f}s"


def ratio(numerator: Optional[float], denominator: Optional[float]) -> str:
    if numerator is None or denominator is None or numerator <= 0:
        return ""
    return f"{denominator / numerator:.2f}x"


def build_prompt(target_tokens: int) -> str:
    return (
        "Write a natural spoken paragraph about Wimbledon and how tennis has changed over the years. "
        "Mention tradition, grass-court tactics, wooden rackets, graphite rackets, serve-and-volley, baseline play, "
        "sports science, prize money, media coverage, Hawk-Eye, and roofed courts. "
        f"Use about {target_tokens} tokens, stay under {target_tokens + 40} tokens, avoid headings and bullets, "
        "and end with a complete sentence."
    )


def iter_sse_events(lines: Iterable[bytes | str]) -> Iterator[tuple[str, str]]:
    event = "message"
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event, "\n".join(data_lines)


def record_openai_text(args, target_tokens: int) -> TextRecording:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    body = {
        "model": args.openai_model,
        "input": build_prompt(target_tokens),
        "max_output_tokens": target_tokens + 80,
        "stream": True,
        "stream_options": {"include_obfuscation": False},
        "store": False,
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
    }
    request = urllib.request.Request(
        args.openai_base_url.rstrip("/") + "/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_token_s: Optional[float] = None
    done_s: Optional[float] = None
    output_tokens: Optional[int] = None
    status: Optional[str] = None
    text_parts: list[str] = []
    delta_events: list[dict] = []
    timeline: list[dict] = [
        {"time_s": 0.0, "source": "openai", "event": "request_start", "target_tokens": target_tokens}
    ]

    def now_s() -> float:
        return time.perf_counter() - start

    try:
        with urllib.request.urlopen(request, timeout=args.openai_timeout) as response:
            for event_name, data_text in iter_sse_events(response):
                now = now_s()
                if data_text == "[DONE]":
                    timeline.append({"time_s": now, "source": "openai", "event": "done_marker"})
                    continue
                data = json.loads(data_text)
                event_type = data.get("type", event_name)
                timeline.append({"time_s": now, "source": "openai", "event": event_type})

                if event_type == "response.output_text.delta":
                    delta = data.get("delta", "")
                    if delta:
                        if first_token_s is None:
                            first_token_s = now
                            timeline.append({"time_s": now, "source": "openai", "event": "first_text_delta"})
                        text_parts.append(delta)
                        delta_events.append({"time_s": now, "delta": delta})
                elif event_type == "response.output_text.done":
                    text = data.get("text")
                    if isinstance(text, str) and text:
                        text_parts = [text]
                elif event_type in {"response.completed", "response.incomplete"}:
                    done_s = now
                    response_obj = data.get("response", {})
                    status = response_obj.get("status")
                    usage = response_obj.get("usage") or {}
                    output_tokens = usage.get("output_tokens")
                    error = response_obj.get("error") or data.get("error")
                    if error:
                        raise RuntimeError(f"OpenAI stream ended with {event_type}: {error}")
                elif event_type == "response.failed" or data.get("error"):
                    raise RuntimeError(f"OpenAI stream failed: {data.get('error')}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc

    if first_token_s is None:
        raise RuntimeError("OpenAI stream produced no text deltas.")
    if done_s is None:
        done_s = now_s()
        timeline.append({"time_s": done_s, "source": "openai", "event": "stream_closed"})

    return TextRecording(
        target_tokens=target_tokens,
        text="".join(text_parts),
        first_token_s=first_token_s,
        done_s=done_s,
        output_tokens=output_tokens,
        status=status,
        delta_events=delta_events,
        timeline=timeline,
    )


def make_delta_replay(recording: TextRecording):
    start = time.perf_counter()

    def replay():
        for event in recording.delta_events:
            target_s = max(0.0, float(event["time_s"]) - recording.first_token_s)
            sleep_s = start + target_s - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            yield event["delta"]

    return replay(), start


def faster_model(args):
    return FasterQwen3TTS.from_pretrained(
        args.custom_model,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        max_seq_len=args.max_seq_len,
    )


def vanilla_model(args):
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        args.custom_model,
        device_map=args.device,
        dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )


def common_generation_kwargs(args, include_chunk_size: bool):
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
    }
    if include_chunk_size:
        kwargs["chunk_size"] = args.chunk_size
    return kwargs


def warm_faster(args, model):
    warm = Namespace(**vars(args))
    warm.max_new_tokens = 24
    warm.min_new_tokens = 1
    warm.chunk_size = 1
    stream = model.generate_custom_voice_streaming(
        text="Wimbledon changed as tennis modernized.",
        speaker=args.speaker,
        language=args.language,
        **common_generation_kwargs(warm, include_chunk_size=True),
    )
    for _ in stream:
        break
    sync_device()


def warm_vanilla(args, model):
    warm = Namespace(**vars(args))
    warm.max_new_tokens = 24
    warm.min_new_tokens = 1
    _ = model.generate_custom_voice(
        text="Wimbledon changed as tennis modernized.",
        speaker=args.speaker,
        language=args.language,
        **common_generation_kwargs(warm, include_chunk_size=False),
    )
    sync_device()


def normalize_audio(audio) -> np.ndarray:
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio).reshape(-1)


def drain_stream(stream_iter, label: str, start_time: float, max_new_tokens: int, chunk_size: int) -> tuple[np.ndarray, TtsMeasurement]:
    chunks = []
    sample_rate = None
    first_audio_s = None
    for chunk_index, item in enumerate(stream_iter):
        chunk, sample_rate = item[:2]
        sync_device()
        now = time.perf_counter() - start_time
        if first_audio_s is None:
            first_audio_s = now
        chunks.append(normalize_audio(chunk))

    sync_device()
    done_s = time.perf_counter() - start_time
    if not chunks:
        raise RuntimeError(f"{label} produced no audio chunks.")
    audio = np.concatenate(chunks)
    measurement = TtsMeasurement(
        label=label,
        first_audio_s=first_audio_s,
        audio_done_s=done_s,
        chunks=len(chunks),
        samples=int(audio.shape[0]),
        sample_rate=int(sample_rate),
        hit_cap=len(chunks) * chunk_size >= max_new_tokens,
    )
    return audio, measurement


def run_faster_text_delta(args, model, recording: TextRecording):
    deltas, start = make_delta_replay(recording)
    stream = model.stream_custom_voice_from_text_deltas(
        text_deltas=deltas,
        speaker=args.speaker,
        language=args.language,
        token_holdback=args.token_holdback,
        **common_generation_kwargs(args, include_chunk_size=True),
    )
    return drain_stream(stream, "faster_text_delta", start, args.max_new_tokens, args.chunk_size)


def run_faster_fulltext(args, model, recording: TextRecording):
    start = time.perf_counter()
    stream = model.generate_custom_voice_streaming(
        text=recording.text,
        speaker=args.speaker,
        language=args.language,
        **common_generation_kwargs(args, include_chunk_size=True),
    )
    audio, measurement = drain_stream(stream, "faster_fulltext", start, args.max_new_tokens, args.chunk_size)
    measurement.first_audio_s += recording.done_after_first_token_s
    measurement.audio_done_s += recording.done_after_first_token_s
    return audio, measurement


def run_vanilla_text_delta(args, model, recording: TextRecording):
    if not hasattr(model, "stream_custom_voice_from_text_deltas"):
        return None, TtsMeasurement(
            label="vanilla_text_delta",
            first_audio_s=None,
            audio_done_s=None,
            available=False,
            note="Qwen3TTSModel has no stream_custom_voice_from_text_deltas method.",
        )
    deltas, start = make_delta_replay(recording)
    stream = model.stream_custom_voice_from_text_deltas(
        text_deltas=deltas,
        speaker=args.speaker,
        language=args.language,
        audio_chunk_code_frames=args.chunk_size,
        token_holdback=args.token_holdback,
        **common_generation_kwargs(args, include_chunk_size=False),
    )
    return drain_stream(stream, "vanilla_text_delta", start, args.max_new_tokens, args.chunk_size)


def run_vanilla_fulltext(args, model, recording: TextRecording):
    start = time.perf_counter()
    audio_list, sample_rate = model.generate_custom_voice(
        text=recording.text,
        speaker=args.speaker,
        language=args.language,
        **common_generation_kwargs(args, include_chunk_size=False),
    )
    sync_device()
    elapsed = time.perf_counter() - start
    audio = normalize_audio(audio_list)
    ready_s = recording.done_after_first_token_s + elapsed
    measurement = TtsMeasurement(
        label="vanilla_fulltext",
        first_audio_s=ready_s,
        audio_done_s=ready_s,
        chunks=1,
        samples=int(audio.shape[0]),
        sample_rate=int(sample_rate),
        hit_cap=None,
        note="Vanilla full-text path returns complete audio; first audio is audio-ready time.",
    )
    return audio, measurement


def write_wav(args, target: int, label: str, audio: Optional[np.ndarray], sample_rate: Optional[int]):
    if not args.write_wavs or audio is None or sample_rate is None:
        return
    path = args.out_dir / "wav" / f"custom_voice_{target}_{label}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def saved(a: TtsMeasurement, b: TtsMeasurement) -> Optional[float]:
    if a.first_audio_s is None or b.first_audio_s is None:
        return None
    return a.first_audio_s - b.first_audio_s


def build_summary_row(args, recording: TextRecording, measurements: dict[str, TtsMeasurement]) -> dict:
    fast_delta = measurements["faster_text_delta"]
    fast_full = measurements["faster_fulltext"]
    vanilla_full = measurements.get("vanilla_fulltext")
    vanilla_delta = measurements.get("vanilla_text_delta")

    row = {
        "target_tokens": recording.target_tokens,
        "openai_model": args.openai_model,
        "openai_output_tokens": recording.output_tokens,
        "openai_deltas": len(recording.delta_events),
        "openai_first_token_from_request_s": metric(recording.first_token_s),
        "llm_done_after_first_token_s": metric(recording.done_after_first_token_s),
        "faster_text_delta_first_audio_after_first_token_s": metric(fast_delta.first_audio_s),
        "faster_fulltext_first_audio_after_first_token_s": metric(fast_full.first_audio_s),
        "faster_text_delta_audio_done_after_first_token_s": metric(fast_delta.audio_done_s),
        "faster_fulltext_audio_done_after_first_token_s": metric(fast_full.audio_done_s),
        "faster_frontside_saved_vs_fulltext_s": metric(saved(fast_full, fast_delta)),
        "faster_text_delta_hit_tts_token_cap": fast_delta.hit_cap,
        "faster_fulltext_hit_tts_token_cap": fast_full.hit_cap,
    }

    if vanilla_full is not None:
        row.update(
            {
                "vanilla_fulltext_audio_ready_after_first_token_s": metric(vanilla_full.first_audio_s),
                "faster_fulltext_backend_saved_vs_vanilla_fulltext_s": metric(saved(vanilla_full, fast_full)),
                "faster_text_delta_total_saved_vs_vanilla_fulltext_s": metric(saved(vanilla_full, fast_delta)),
            }
        )
    if vanilla_delta is not None:
        row.update(
            {
                "vanilla_text_delta_first_audio_after_first_token_s": metric(vanilla_delta.first_audio_s),
                "vanilla_frontside_saved_vs_fulltext_s": metric(saved(vanilla_full, vanilla_delta) if vanilla_full else None),
                "faster_text_delta_backend_saved_vs_vanilla_text_delta_s": metric(saved(vanilla_delta, fast_delta)),
                "vanilla_text_delta_hit_tts_token_cap": vanilla_delta.hit_cap,
                "vanilla_text_delta_available": vanilla_delta.available,
            }
        )
    return row


def markdown_tables(rows: list[dict]) -> str:
    lines = [
        "### Normalized LLM-to-TTS Timeline",
        "",
        "`T+0.000s` is the first LLM text delta. ",
        "🟩 means at least 1s saved, 🟨 means 0.25-1s saved, and ⬜ means under 0.25s saved.",
        "",
        "| Target | This repo: Faster + text-delta first audio | LLM done | Faster + full-text first audio | Vanilla Qwen full-text audio ready |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {fast_delta} | {llm_done} | {fast_full} | {vanilla_full} |".format(
                target=f"{row['target_tokens']} tokens",
                fast_delta=tplus(_float(row.get("faster_text_delta_first_audio_after_first_token_s"))),
                llm_done=tplus(_float(row.get("llm_done_after_first_token_s"))),
                fast_full=tplus(_float(row.get("faster_fulltext_first_audio_after_first_token_s"))),
                vanilla_full=tplus(_float(row.get("vanilla_fulltext_audio_ready_after_first_token_s"))),
            )
        )

    lines.extend(
        [
            "",
            "### Improvement Breakdown",
            "",
            "| Target | Backend gain: Faster full-text vs vanilla full-text | Front-side gain: Faster text-delta vs Faster full-text | Total gain: this repo vs vanilla full-text |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        total = _float(row.get("faster_text_delta_total_saved_vs_vanilla_fulltext_s"))
        fast_delta = _float(row.get("faster_text_delta_first_audio_after_first_token_s"))
        vanilla_full = _float(row.get("vanilla_fulltext_audio_ready_after_first_token_s"))
        lines.append(
            "| {target} | {backend_full} | {front_fast} | {total} ({ratio}) |".format(
                target=f"{row['target_tokens']} tokens",
                backend_full=improvement(_float(row.get("faster_fulltext_backend_saved_vs_vanilla_fulltext_s"))),
                front_fast=improvement(_float(row.get("faster_frontside_saved_vs_fulltext_s"))),
                total=improvement(total),
                ratio=ratio(fast_delta, vanilla_full),
            )
        )
    return "\n".join(lines) + "\n"


def _float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def write_recordings(args, recordings: list[TextRecording]):
    records_dir = args.out_dir / "recordings"
    records_dir.mkdir(parents=True, exist_ok=True)
    for recording in recordings:
        payload = {
            "target_tokens": recording.target_tokens,
            "text": recording.text,
            "first_token_s": recording.first_token_s,
            "done_s": recording.done_s,
            "done_after_first_token_s": recording.done_after_first_token_s,
            "output_tokens": recording.output_tokens,
            "status": recording.status,
            "delta_events": recording.delta_events,
            "timeline": recording.timeline,
        }
        path = records_dir / f"openai_{recording.target_tokens}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "openai_model": args.openai_model,
        "custom_model": args.custom_model,
        "targets": args.targets,
        "speaker": args.speaker,
        "chunk_size": args.chunk_size,
        "token_holdback": args.token_holdback,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "device": args.device,
        "dtype": args.dtype,
        "normalization": "All downstream TTS times are seconds after the first OpenAI response.output_text.delta.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Recording OpenAI streams...", flush=True)
    recordings = [record_openai_text(args, target) for target in args.targets]
    write_recordings(args, recordings)

    rows: list[dict] = []
    measurements_by_target: dict[int, dict[str, TtsMeasurement]] = {
        recording.target_tokens: {} for recording in recordings
    }

    print("Loading FasterQwen3TTS...", flush=True)
    fast = faster_model(args)
    warm_faster(args, fast)
    for recording in recordings:
        target = recording.target_tokens
        print(f"Faster target={target}: text-delta", flush=True)
        audio, measurement = run_faster_text_delta(args, fast, recording)
        measurements_by_target[target]["faster_text_delta"] = measurement
        write_wav(args, target, measurement.label, audio, measurement.sample_rate)

        print(f"Faster target={target}: full-text", flush=True)
        audio, measurement = run_faster_fulltext(args, fast, recording)
        measurements_by_target[target]["faster_fulltext"] = measurement
        write_wav(args, target, measurement.label, audio, measurement.sample_rate)
    del fast
    sync_device()
    torch.cuda.empty_cache()

    if not args.skip_vanilla:
        print("Loading vanilla Qwen3TTSModel...", flush=True)
        vanilla = vanilla_model(args)
        warm_vanilla(args, vanilla)
        for recording in recordings:
            target = recording.target_tokens
            if args.include_vanilla_text_delta:
                print(f"Vanilla target={target}: text-delta", flush=True)
                audio, measurement = run_vanilla_text_delta(args, vanilla, recording)
                measurements_by_target[target]["vanilla_text_delta"] = measurement
                write_wav(args, target, measurement.label, audio, measurement.sample_rate)

            print(f"Vanilla target={target}: full-text", flush=True)
            audio, measurement = run_vanilla_fulltext(args, vanilla, recording)
            measurements_by_target[target]["vanilla_fulltext"] = measurement
            write_wav(args, target, measurement.label, audio, measurement.sample_rate)
        del vanilla
        sync_device()
        torch.cuda.empty_cache()

    for recording in recordings:
        rows.append(build_summary_row(args, recording, measurements_by_target[recording.target_tokens]))

    fieldnames = sorted({key for row in rows for key in row})
    with (args.out_dir / "summary_normalized.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table = markdown_tables(rows)
    (args.out_dir / "readme_tables.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote normalized benchmark output to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
