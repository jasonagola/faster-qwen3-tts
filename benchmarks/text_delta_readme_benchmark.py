#!/usr/bin/env python3
"""Generate README timing data and curated text-delta audio samples.

This script is intentionally manual and env-gated. It writes benchmark CSV,
timeline JSONL, and generated WAV files under ``--out-dir``; commit only the
curated sample files you want to publish.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import threading
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


DEFAULT_REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
DEFAULT_REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


@dataclass
class OpenAIResult:
    target_tokens: int
    text: str
    first_token_s: Optional[float]
    done_s: float
    output_tokens: Optional[int]
    status: Optional[str]
    deltas: int
    timeline: list[dict]

    @property
    def generation_tokens_per_s(self) -> Optional[float]:
        if self.output_tokens is None or self.first_token_s is None:
            return None
        span = self.done_s - self.first_token_s
        if span <= 0:
            return None
        return self.output_tokens / span


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small text-delta input streaming benchmark for README data.")
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-timeout", type=float, default=180.0)
    parser.add_argument("--targets", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--custom-model", default=DEFAULT_CUSTOM_MODEL)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--voice-design-model", default=DEFAULT_VOICE_DESIGN_MODEL)
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--language", default="English")
    parser.add_argument("--voice-design-instruct", default="Speak clearly in a calm, neutral adult voice.")
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
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
    parser.add_argument("--out-dir", type=Path, default=Path("text_delta_readme_benchmark"))
    parser.add_argument("--samples-dir", type=Path, default=Path("samples/text_delta_streaming"))
    parser.add_argument("--sample-max-new-tokens", type=int, default=1536)
    parser.add_argument("--skip-samples", action="store_true")
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


def row_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


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


def start_openai_text_source(args, target_tokens: int):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    sentinel = object()
    pending: queue.Queue[object] = queue.Queue()
    state = {
        "start_time": time.perf_counter(),
        "text_parts": [],
        "first_token_s": None,
        "done_s": None,
        "output_tokens": None,
        "status": None,
        "deltas": 0,
        "error": None,
        "timeline": [],
    }
    prompt = build_prompt(target_tokens)
    body = {
        "model": args.openai_model,
        "input": prompt,
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
    state["timeline"].append({"time_s": 0.0, "source": "openai", "event": "request_start", "target_tokens": target_tokens})

    def now_s() -> float:
        return time.perf_counter() - state["start_time"]

    def produce():
        try:
            with urllib.request.urlopen(request, timeout=args.openai_timeout) as response:
                for event_name, data_text in iter_sse_events(response):
                    now = now_s()
                    if data_text == "[DONE]":
                        state["timeline"].append({"time_s": now, "source": "openai", "event": "done_marker"})
                        continue
                    data = json.loads(data_text)
                    event_type = data.get("type", event_name)
                    state["timeline"].append({"time_s": now, "source": "openai", "event": event_type})

                    if event_type == "response.output_text.delta":
                        delta = data.get("delta", "")
                        if delta:
                            if state["first_token_s"] is None:
                                state["first_token_s"] = now
                                state["timeline"].append({"time_s": now, "source": "openai", "event": "first_text_delta"})
                            state["text_parts"].append(delta)
                            state["deltas"] += 1
                            pending.put(delta)
                    elif event_type == "response.output_text.done":
                        text = data.get("text")
                        if isinstance(text, str) and text:
                            state["text_parts"] = [text]
                    elif event_type in {"response.completed", "response.incomplete"}:
                        state["done_s"] = now
                        response_obj = data.get("response", {})
                        state["status"] = response_obj.get("status")
                        usage = response_obj.get("usage") or {}
                        state["output_tokens"] = usage.get("output_tokens")
                        error = response_obj.get("error") or data.get("error")
                        if error:
                            raise RuntimeError(f"OpenAI stream ended with {event_type}: {error}")
                    elif event_type == "response.failed" or data.get("error"):
                        raise RuntimeError(f"OpenAI stream failed: {data.get('error')}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            state["error"] = RuntimeError(f"OpenAI HTTP {exc.code}: {detail}")
        except BaseException as exc:  # noqa: BLE001 - forward producer failures to consumer.
            state["error"] = exc
        finally:
            if state["done_s"] is None:
                state["done_s"] = now_s()
                state["timeline"].append({"time_s": state["done_s"], "source": "openai", "event": "stream_closed"})
            pending.put(sentinel)

    def consume():
        while True:
            item = pending.get()
            if item is sentinel:
                break
            yield item
        if state["error"] is not None:
            raise state["error"]

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    return consume(), thread, state


def openai_result_from_state(target_tokens: int, state: dict) -> OpenAIResult:
    return OpenAIResult(
        target_tokens=target_tokens,
        text="".join(state["text_parts"]),
        first_token_s=state["first_token_s"],
        done_s=float(state["done_s"] or 0.0),
        output_tokens=state["output_tokens"],
        status=state["status"],
        deltas=int(state["deltas"]),
        timeline=list(state["timeline"]),
    )


def load_model(args, model_id: str) -> FasterQwen3TTS:
    return FasterQwen3TTS.from_pretrained(
        model_id,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        max_seq_len=args.max_seq_len,
    )


def generation_kwargs(args, max_new_tokens: Optional[int] = None):
    return {
        "max_new_tokens": max_new_tokens or args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "chunk_size": args.chunk_size,
    }


def warm_model(args, model, mode: str):
    warm_args = Namespace(**vars(args))
    warm_args.max_new_tokens = 24
    warm_args.min_new_tokens = 1
    warm_args.chunk_size = 1
    text = "Wimbledon changed as tennis modernized."
    stream = fulltext_stream(model, mode, text, warm_args)
    for _ in stream:
        break
    sync_device()


def fulltext_stream(model, mode: str, text: str, args):
    common = generation_kwargs(args)
    if mode == "custom_voice":
        return model.generate_custom_voice_streaming(text, speaker=args.speaker, language=args.language, **common)
    if mode == "voice_design":
        return model.generate_voice_design_streaming(text, instruct=args.voice_design_instruct, language=args.language, **common)
    if mode == "voice_clone_xvec":
        return model.generate_voice_clone_streaming(
            text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=True,
            **common,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def text_delta_stream(model, mode: str, text_deltas: Iterable[str], args):
    common = generation_kwargs(args)
    common["token_holdback"] = args.token_holdback
    if mode == "custom_voice":
        return model.stream_custom_voice_from_text_deltas(text_deltas, speaker=args.speaker, language=args.language, **common)
    if mode == "voice_design":
        return model.stream_voice_design_from_text_deltas(text_deltas, instruct=args.voice_design_instruct, language=args.language, **common)
    if mode == "voice_clone_xvec":
        return model.stream_voice_clone_from_text_deltas(
            text_deltas,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=True,
            **common,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def drain_audio(stream_iter, timeline: list[dict], source: str, start_time: float, offset_s: float = 0.0):
    chunks = []
    sr = None
    first_s = None
    chunk_count = 0
    for item in stream_iter:
        chunk, sr = item[:2]
        sync_device()
        now = offset_s + (time.perf_counter() - start_time)
        if first_s is None:
            first_s = now
            timeline.append({"time_s": now, "source": source, "event": "first_audio"})
        timeline.append({"time_s": now, "source": source, "event": "audio_chunk", "chunk_index": chunk_count, "samples": int(len(chunk))})
        chunks.append(chunk)
        chunk_count += 1
    sync_device()
    done_s = offset_s + (time.perf_counter() - start_time)
    timeline.append({"time_s": done_s, "source": source, "event": "audio_done", "chunks": chunk_count})
    if not chunks:
        raise RuntimeError(f"{source} produced no audio chunks")
    return np.concatenate(chunks), int(sr), first_s, done_s, chunk_count


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def validate_wav(path: Path, min_duration_s: float = 0.25, max_duration_s: float = 180.0):
    if not path.exists():
        raise RuntimeError(f"Missing WAV file: {path}")
    audio, sr = sf.read(path, always_2d=False)
    samples = np.asarray(audio)
    if sr <= 0:
        raise RuntimeError(f"Invalid sample rate for {path}: {sr}")
    if samples.size == 0:
        raise RuntimeError(f"Empty WAV file: {path}")
    if not np.isfinite(samples).all():
        raise RuntimeError(f"WAV contains NaN/inf samples: {path}")
    duration_s = samples.shape[0] / sr
    if duration_s < min_duration_s or duration_s > max_duration_s:
        raise RuntimeError(f"WAV duration out of bounds for {path}: {duration_s:.3f}s")
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-4:
        raise RuntimeError(f"WAV appears silent: {path}")


def validate_sample_readme_links(samples_dir: Path, filenames: Iterable[str]):
    readme = samples_dir / "README.md"
    text = readme.read_text(encoding="utf-8")
    for filename in filenames:
        if filename not in text:
            raise RuntimeError(f"{readme} does not reference {filename}")


def split_text_for_samples(text: str, words_per_delta: int = 4) -> list[str]:
    words = text.split()
    if not words:
        return []
    deltas = []
    for index in range(0, len(words), words_per_delta):
        chunk = " ".join(words[index:index + words_per_delta])
        if index + words_per_delta < len(words):
            chunk += " "
        deltas.append(chunk)
    return deltas


def run_custom_voice_benchmark(args, writer) -> dict[int, str]:
    model = load_model(args, args.custom_model)
    sync_device()
    warm_start = time.perf_counter()
    warm_model(args, model, "custom_voice")
    warm_s = time.perf_counter() - warm_start

    target_texts: dict[int, str] = {}
    for target in args.targets:
        timeline: list[dict] = [{"time_s": 0.0, "source": "tts", "event": "model_hot"}]
        text_source, openai_thread, openai_state = start_openai_text_source(args, target)
        t0 = openai_state["start_time"]
        delta_audio, delta_sr, delta_first, delta_done, delta_chunks = drain_audio(
            text_delta_stream(model, "custom_voice", text_source, args),
            timeline,
            "text_delta",
            t0,
        )
        openai_thread.join()
        if openai_state["error"] is not None:
            raise openai_state["error"]
        result = openai_result_from_state(target, openai_state)
        timeline.extend(result.timeline)
        target_texts[target] = result.text

        full_start = time.perf_counter()
        full_audio, full_sr, full_first, full_done, full_chunks = drain_audio(
            fulltext_stream(model, "custom_voice", result.text, args),
            timeline,
            "fulltext",
            full_start,
            offset_s=result.done_s,
        )

        stem = f"custom_voice_{target}"
        write_wav(args.out_dir / "wav" / f"{stem}_text_delta.wav", delta_audio, delta_sr)
        write_wav(args.out_dir / "wav" / f"{stem}_fulltext.wav", full_audio, full_sr)
        write_jsonl(args.out_dir / "timelines" / f"{stem}.jsonl", sorted(timeline, key=lambda item: item.get("time_s", 0.0)))
        (args.out_dir / "texts").mkdir(parents=True, exist_ok=True)
        (args.out_dir / "texts" / f"{stem}.txt").write_text(result.text, encoding="utf-8")

        delta_hit_cap = delta_chunks * args.chunk_size >= args.max_new_tokens
        full_hit_cap = full_chunks * args.chunk_size >= args.max_new_tokens
        writer.writerow(
            {
                "target_tokens": target,
                "openai_model": args.openai_model,
                "tts_model": args.custom_model,
                "mode": "custom_voice",
                "openai_status": result.status,
                "openai_output_tokens": result.output_tokens,
                "openai_deltas": result.deltas,
                "openai_first_token_s": row_value(result.first_token_s),
                "openai_done_s": row_value(result.done_s),
                "openai_generation_tokens_per_s": row_value(result.generation_tokens_per_s),
                "text_delta_first_audio_s": row_value(delta_first),
                "fulltext_first_audio_s": row_value(full_first),
                "first_token_to_audio_s": row_value(delta_first - result.first_token_s if result.first_token_s is not None else None),
                "audio_before_llm_done_s": row_value(result.done_s - delta_first if delta_first is not None else None),
                "text_delta_audio_done_s": row_value(delta_done),
                "fulltext_audio_done_s": row_value(full_done),
                "text_delta_chunks": delta_chunks,
                "fulltext_chunks": full_chunks,
                "text_delta_hit_tts_token_cap": delta_hit_cap,
                "fulltext_hit_tts_token_cap": full_hit_cap,
                "tts_warmup_s": row_value(warm_s),
            }
        )
        print(
            f"custom_voice target={target} first_token={row_value(result.first_token_s)}s "
            f"text_delta_first_audio={row_value(delta_first)}s fulltext_first_audio={row_value(full_first)}s "
            f"caps={delta_hit_cap}/{full_hit_cap}",
            flush=True,
        )

    del model
    sync_device()
    torch.cuda.empty_cache()
    return target_texts


def generate_sample_pair(args, model, mode: str, text: str, output_stem: str):
    sample_args = Namespace(**vars(args))
    sample_args.max_new_tokens = args.sample_max_new_tokens
    deltas = split_text_for_samples(text)
    delta_audio, delta_sr, _, _, delta_chunks = drain_audio(
        text_delta_stream(model, mode, deltas, sample_args),
        [],
        f"{output_stem}:text_delta",
        time.perf_counter(),
    )
    full_audio, full_sr, _, _, full_chunks = drain_audio(
        fulltext_stream(model, mode, text, sample_args),
        [],
        f"{output_stem}:fulltext",
        time.perf_counter(),
    )
    if delta_chunks * sample_args.chunk_size >= sample_args.max_new_tokens:
        raise RuntimeError(f"{output_stem} text-delta sample hit max_new_tokens={sample_args.max_new_tokens}")
    if full_chunks * sample_args.chunk_size >= sample_args.max_new_tokens:
        raise RuntimeError(f"{output_stem} full-text sample hit max_new_tokens={sample_args.max_new_tokens}")
    text_delta_path = args.samples_dir / f"{output_stem}_text_delta.wav"
    fulltext_path = args.samples_dir / f"{output_stem}_fulltext.wav"
    write_wav(text_delta_path, delta_audio, delta_sr)
    write_wav(fulltext_path, full_audio, full_sr)
    validate_wav(text_delta_path)
    validate_wav(fulltext_path)


def generate_curated_samples(args, target_texts: dict[int, str]):
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    sample_text_100 = target_texts.get(100) or (
        "Wimbledon has kept its old rituals while tennis has become faster, stronger, and more technical."
    )
    sample_text_200 = target_texts.get(200) or sample_text_100

    custom = load_model(args, args.custom_model)
    warm_model(args, custom, "custom_voice")
    generate_sample_pair(args, custom, "custom_voice", sample_text_200, "custom_voice_200")
    del custom
    sync_device()
    torch.cuda.empty_cache()

    design = load_model(args, args.voice_design_model)
    warm_model(args, design, "voice_design")
    generate_sample_pair(args, design, "voice_design", sample_text_100, "voice_design_100")
    del design
    sync_device()
    torch.cuda.empty_cache()

    clone = load_model(args, args.base_model)
    warm_model(args, clone, "voice_clone_xvec")
    generate_sample_pair(args, clone, "voice_clone_xvec", sample_text_100, "voice_clone_xvec_100")
    del clone
    sync_device()
    torch.cuda.empty_cache()

    readme = args.samples_dir / "README.md"
    readme.write_text(
        "# Text-Delta Streaming Samples\n\n"
        "These samples compare the same text through text-delta input streaming and full-text audio streaming. "
        "Each pair uses the same prompt and generation settings except for the input path. "
        "They are generated by `benchmarks/text_delta_readme_benchmark.py` and validated for nonzero duration, "
        "finite samples, and non-silent peak amplitude.\n\n"
        "Default sample settings: `chunk_size=8`, `token_holdback=1`, `do_sample=True`, "
        "`temperature=0.9`, `top_k=50`, `top_p=1.0`, and `repetition_penalty=1.05`.\n\n"
        "| Mode | Text-delta input | Full-text input |\n"
        "|---|---|---|\n"
        "| CustomVoice 200-token sample | `custom_voice_200_text_delta.wav` | `custom_voice_200_fulltext.wav` |\n"
        "| VoiceDesign 100-token sample | `voice_design_100_text_delta.wav` | `voice_design_100_fulltext.wav` |\n"
        "| Voice clone x-vector 100-token sample | `voice_clone_xvec_100_text_delta.wav` | `voice_clone_xvec_100_fulltext.wav` |\n",
        encoding="utf-8",
    )
    validate_sample_readme_links(
        args.samples_dir,
        [
            "custom_voice_200_text_delta.wav",
            "custom_voice_200_fulltext.wav",
            "voice_design_100_text_delta.wav",
            "voice_design_100_fulltext.wav",
            "voice_clone_xvec_100_text_delta.wav",
            "voice_clone_xvec_100_fulltext.wav",
        ],
    )


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
        "base_model": args.base_model,
        "voice_design_model": args.voice_design_model,
        "targets": args.targets,
        "chunk_size": args.chunk_size,
        "token_holdback": args.token_holdback,
        "max_new_tokens": args.max_new_tokens,
        "sample_max_new_tokens": args.sample_max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "device": args.device,
        "dtype": args.dtype,
        "tts_warmup_excluded_from_request_timing": True,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    fields = [
        "target_tokens",
        "openai_model",
        "tts_model",
        "mode",
        "openai_status",
        "openai_output_tokens",
        "openai_deltas",
        "openai_first_token_s",
        "openai_done_s",
        "openai_generation_tokens_per_s",
        "text_delta_first_audio_s",
        "fulltext_first_audio_s",
        "first_token_to_audio_s",
        "audio_before_llm_done_s",
        "text_delta_audio_done_s",
        "fulltext_audio_done_s",
        "text_delta_chunks",
        "fulltext_chunks",
        "text_delta_hit_tts_token_cap",
        "fulltext_hit_tts_token_cap",
        "tts_warmup_s",
    ]
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        target_texts = run_custom_voice_benchmark(args, writer)

    if not args.skip_samples:
        generate_curated_samples(args, target_texts)

    print(f"Wrote benchmark output to {args.out_dir}")
    if not args.skip_samples:
        print(f"Wrote curated samples to {args.samples_dir}")


if __name__ == "__main__":
    main()
