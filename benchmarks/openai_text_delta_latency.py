#!/usr/bin/env python3
"""Benchmark a live OpenAI text stream feeding Qwen3-TTS in real time.

Each engine/mode/token-limit case opens a fresh streamed OpenAI Responses API
request. Text deltas are queued into the TTS generator immediately as they
arrive, while the same deltas are also collected for the full-text baseline
that starts only after the LLM stream has completed.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import soundfile as sf
import torch


THIS_REPO = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_REPO))
sys.path.insert(0, str(BENCHMARKS_DIR))

from compare_text_delta_input import (  # noqa: E402
    DEFAULT_REF_AUDIO,
    DEFAULT_REF_TEXT,
    DEFAULT_UPSTREAM_REPO,
    count_tokens,
    fulltext_stream,
    load_engine,
    sync_device,
    text_delta_stream,
)


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OUTPUT_LIMITS = [100, 200, 500, 1000]


@dataclass
class RecordedDelta:
    index: int
    offset_s: float
    delta: str


@dataclass
class OpenAIStreamResult:
    model: str
    output_limit: int
    prompt: str
    text: str
    deltas: list[RecordedDelta]
    first_token_s: Optional[float]
    completed_s: float
    usage: dict
    response_id: Optional[str]
    status: Optional[str]
    timeline: list[dict]

    @property
    def output_tokens(self) -> Optional[int]:
        value = self.usage.get("output_tokens")
        return value if isinstance(value, int) else None

    @property
    def request_tokens_per_second(self) -> Optional[float]:
        if self.output_tokens is None or self.completed_s <= 0:
            return None
        return self.output_tokens / self.completed_s

    @property
    def generation_tokens_per_second(self) -> Optional[float]:
        if self.output_tokens is None or self.first_token_s is None:
            return None
        span = self.completed_s - self.first_token_s
        if span <= 0:
            return None
        return self.output_tokens / span


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure real OpenAI request streaming into Qwen3-TTS text-delta input streaming."
    )
    parser.add_argument("--upstream-repo", type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--engines", nargs="+", choices=["upstream", "faster"], default=["upstream", "faster"])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["custom_voice", "voice_design", "voice_clone_xvec", "voice_clone_icl"],
        default=["custom_voice"],
    )
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-timeout", type=float, default=180.0)
    parser.add_argument("--openai-reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="none")
    parser.add_argument("--openai-text-verbosity", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--openai-service-tier", default=None)
    parser.add_argument("--output-limits", nargs="+", type=int, default=DEFAULT_OUTPUT_LIMITS)
    parser.add_argument("--openai-temperature", type=float, default=None)
    parser.add_argument("--openai-top-p", type=float, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--custom-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--base-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--voice-design-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--instruct", default="")
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
    parser.add_argument("--decode-left-context-frames", type=int, default=25)
    parser.add_argument("--token-holdback", type=int, default=1)
    parser.add_argument("--tts-do-sample", action="store_true")
    parser.add_argument("--tts-temperature", type=float, default=0.9)
    parser.add_argument("--tts-top-k", type=int, default=50)
    parser.add_argument("--tts-top-p", type=float, default=1.0)
    parser.add_argument("--tts-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--out-dir", type=Path, default=Path("openai_text_delta_latency"))
    parser.add_argument("--write-wavs", action="store_true")
    parser.add_argument("--print-openai-deltas", action="store_true")
    parser.add_argument("--no-tts-warmup", action="store_true", help="Include first-use TTS setup in request timing.")
    args = parser.parse_args()
    args.do_sample = args.tts_do_sample
    args.temperature = args.tts_temperature
    args.top_k = args.tts_top_k
    args.top_p = args.tts_top_p
    args.repetition_penalty = args.tts_repetition_penalty
    return args


def build_wimbledon_prompt(output_limit: int, override: Optional[str] = None) -> str:
    if override:
        return override.format(output_limit=output_limit)
    return (
        "Write a natural spoken explanation about Wimbledon and how tennis has changed over the years. "
        "Cover tradition, grass-court tactics, wooden rackets versus graphite, serve-and-volley versus baseline "
        "play, athleticism, sports science, prize money, media coverage, Hawk-Eye, roofed courts, and how the event "
        "has kept its identity while the sport around it modernized. Do not use headings, bullets, markdown, or "
        "stage directions. Keep the prose easy for text-to-speech. Stop cleanly before the limit. "
        f"Hard cap: {output_limit} output tokens."
    )


def iter_sse_events_from_lines(lines: Iterable[bytes | str]) -> Iterator[tuple[str, str]]:
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


def build_openai_request_body(args, output_limit: int, prompt: str) -> dict:
    body = {
        "model": args.openai_model,
        "input": prompt,
        "max_output_tokens": output_limit,
        "stream": True,
        "stream_options": {"include_obfuscation": False},
        "store": False,
        "text": {"verbosity": args.openai_text_verbosity},
    }
    if args.openai_reasoning_effort:
        body["reasoning"] = {"effort": args.openai_reasoning_effort}
    if args.openai_service_tier:
        body["service_tier"] = args.openai_service_tier
    if args.openai_temperature is not None:
        body["temperature"] = args.openai_temperature
    if args.openai_top_p is not None:
        body["top_p"] = args.openai_top_p
    return body


def stream_openai_response(args, output_limit: int, prompt: str) -> OpenAIStreamResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; export it before running this live benchmark.")

    url = args.openai_base_url.rstrip("/") + "/responses"
    body = build_openai_request_body(args, output_limit, prompt)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    deltas: list[RecordedDelta] = []
    timeline: list[dict] = [
        {
            "time_s": 0.0,
            "source": "openai",
            "event": "request_start",
            "model": args.openai_model,
            "output_limit": output_limit,
        }
    ]
    text_parts: list[str] = []
    usage: dict = {}
    response_id = None
    status = None
    first_token_s = None
    completed_s = None

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.openai_timeout) as response:
            for event_name, data_text in iter_sse_events_from_lines(response):
                now = time.perf_counter() - t0
                if data_text == "[DONE]":
                    timeline.append({"time_s": now, "source": "openai", "event": "done_marker"})
                    continue
                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Could not parse OpenAI SSE payload for {event_name}: {data_text[:200]}") from exc

                event_type = data.get("type", event_name)
                timeline.append({"time_s": now, "source": "openai", "event": event_type})

                if event_type == "response.output_text.delta":
                    delta = data.get("delta", "")
                    if delta:
                        if first_token_s is None:
                            first_token_s = now
                            timeline.append({"time_s": now, "source": "openai", "event": "first_text_delta"})
                        deltas.append(RecordedDelta(len(deltas), now, delta))
                        text_parts.append(delta)
                        if args.print_openai_deltas:
                            print(delta, end="", flush=True)
                elif event_type == "response.output_text.done":
                    done_text = data.get("text")
                    if isinstance(done_text, str) and done_text:
                        text_parts = [done_text]
                elif event_type == "response.completed":
                    completed_s = now
                    response_obj = data.get("response", {})
                    response_id = response_obj.get("id")
                    status = response_obj.get("status")
                    usage = response_obj.get("usage") or {}
                elif event_type in {"response.failed", "response.incomplete"}:
                    completed_s = now
                    response_obj = data.get("response", {})
                    response_id = response_obj.get("id")
                    status = response_obj.get("status")
                    usage = response_obj.get("usage") or {}
                    error = response_obj.get("error") or data.get("error")
                    incomplete_details = response_obj.get("incomplete_details") or {}
                    incomplete_reason = incomplete_details.get("reason")
                    if event_type == "response.failed" or error:
                        raise RuntimeError(f"OpenAI stream ended with {event_type}: {error}")
                    if event_type == "response.incomplete" and incomplete_reason not in {None, "max_output_tokens"}:
                        raise RuntimeError(f"OpenAI stream ended incomplete: {incomplete_details}")
                elif "error" in data and data.get("error"):
                    raise RuntimeError(f"OpenAI stream error: {data['error']}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc

    if completed_s is None:
        completed_s = time.perf_counter() - t0
        timeline.append({"time_s": completed_s, "source": "openai", "event": "stream_closed"})
    if args.print_openai_deltas:
        print()

    return OpenAIStreamResult(
        model=args.openai_model,
        output_limit=output_limit,
        prompt=prompt,
        text="".join(text_parts),
        deltas=deltas,
        first_token_s=first_token_s,
        completed_s=completed_s,
        usage=usage,
        response_id=response_id,
        status=status,
        timeline=timeline,
    )


def start_live_openai_delta_source(args, output_limit: int, prompt: str, timeline: list[dict]):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; export it before running this live benchmark.")

    sentinel = object()
    pending: queue.Queue[object] = queue.Queue()
    state = {
        "start_time": time.perf_counter(),
        "text_parts": [],
        "deltas": [],
        "first_token_s": None,
        "completed_s": None,
        "usage": {},
        "response_id": None,
        "status": None,
        "error": None,
    }

    url = args.openai_base_url.rstrip("/") + "/responses"
    body = build_openai_request_body(args, output_limit, prompt)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    timeline.append(
        {
            "time_s": 0.0,
            "source": "openai",
            "event": "request_start",
            "model": args.openai_model,
            "output_limit": output_limit,
        }
    )

    def now_s() -> float:
        return time.perf_counter() - state["start_time"]

    def produce():
        try:
            with urllib.request.urlopen(request, timeout=args.openai_timeout) as response:
                for event_name, data_text in iter_sse_events_from_lines(response):
                    now = now_s()
                    if data_text == "[DONE]":
                        timeline.append({"time_s": now, "source": "openai", "event": "done_marker"})
                        continue
                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Could not parse OpenAI SSE payload for {event_name}: {data_text[:200]}"
                        ) from exc

                    event_type = data.get("type", event_name)
                    timeline.append({"time_s": now, "source": "openai", "event": event_type})

                    if event_type == "response.output_text.delta":
                        delta = data.get("delta", "")
                        if delta:
                            if state["first_token_s"] is None:
                                state["first_token_s"] = now
                                timeline.append({"time_s": now, "source": "openai", "event": "first_text_delta"})
                            recorded = RecordedDelta(len(state["deltas"]), now, delta)
                            state["deltas"].append(recorded)
                            state["text_parts"].append(delta)
                            pending.put(delta)
                            if args.print_openai_deltas:
                                print(delta, end="", flush=True)
                    elif event_type == "response.output_text.done":
                        done_text = data.get("text")
                        if isinstance(done_text, str) and done_text:
                            state["text_parts"] = [done_text]
                    elif event_type == "response.completed":
                        state["completed_s"] = now
                        response_obj = data.get("response", {})
                        state["response_id"] = response_obj.get("id")
                        state["status"] = response_obj.get("status")
                        state["usage"] = response_obj.get("usage") or {}
                    elif event_type in {"response.failed", "response.incomplete"}:
                        state["completed_s"] = now
                        response_obj = data.get("response", {})
                        state["response_id"] = response_obj.get("id")
                        state["status"] = response_obj.get("status")
                        state["usage"] = response_obj.get("usage") or {}
                        error = response_obj.get("error") or data.get("error")
                        incomplete_details = response_obj.get("incomplete_details") or {}
                        incomplete_reason = incomplete_details.get("reason")
                        if event_type == "response.failed" or error:
                            raise RuntimeError(f"OpenAI stream ended with {event_type}: {error}")
                        if event_type == "response.incomplete" and incomplete_reason not in {None, "max_output_tokens"}:
                            raise RuntimeError(f"OpenAI stream ended incomplete: {incomplete_details}")
                    elif "error" in data and data.get("error"):
                        raise RuntimeError(f"OpenAI stream error: {data['error']}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            state["error"] = RuntimeError(f"OpenAI HTTP {exc.code}: {detail}")
        except BaseException as exc:  # noqa: BLE001 - preserve producer failure for consumer thread.
            state["error"] = exc
        finally:
            if state["completed_s"] is None:
                state["completed_s"] = now_s()
                timeline.append({"time_s": state["completed_s"], "source": "openai", "event": "stream_closed"})
            if args.print_openai_deltas:
                print()
            pending.put(sentinel)

    def consume():
        while True:
            item = pending.get()
            if item is sentinel:
                break
            yield item
        if state["error"] is not None:
            raise state["error"]

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    return consume(), producer, state


def live_state_to_result(args, output_limit: int, prompt: str, state: dict, timeline: list[dict]) -> OpenAIStreamResult:
    return OpenAIStreamResult(
        model=args.openai_model,
        output_limit=output_limit,
        prompt=prompt,
        text="".join(state["text_parts"]),
        deltas=list(state["deltas"]),
        first_token_s=state["first_token_s"],
        completed_s=state["completed_s"] or 0.0,
        usage=state["usage"],
        response_id=state["response_id"],
        status=state["status"],
        timeline=timeline,
    )


def start_recorded_delta_replay(result: OpenAIStreamResult):
    sentinel = object()
    pending: queue.Queue[object] = queue.Queue()
    state = {
        "start_time": time.perf_counter(),
        "emitted_deltas": 0,
        "llm_done_s": None,
    }

    def sleep_until(target_offset_s: float):
        while True:
            remaining = target_offset_s - (time.perf_counter() - state["start_time"])
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.01))

    def produce():
        for recorded in result.deltas:
            sleep_until(recorded.offset_s)
            state["emitted_deltas"] += 1
            pending.put(recorded.delta)
        sleep_until(result.completed_s)
        state["llm_done_s"] = time.perf_counter() - state["start_time"]
        pending.put(sentinel)

    def consume():
        while True:
            item = pending.get()
            if item is sentinel:
                break
            yield item

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    return consume(), thread, state


def drain_audio_stream(stream_iter, timeline: list[dict], source: str, start_time: float, offset_s: float = 0.0):
    chunks = []
    stream_sr = None
    first_audio_s = None
    chunk_count = 0
    for item in stream_iter:
        wav, stream_sr = item[:2]
        sync_device()
        now = offset_s + (time.perf_counter() - start_time)
        if first_audio_s is None:
            first_audio_s = now
            timeline.append({"time_s": now, "source": source, "event": "first_audio"})
        chunk_count += 1
        timeline.append(
            {
                "time_s": now,
                "source": source,
                "event": "audio_chunk",
                "chunk_index": chunk_count - 1,
                "samples": int(len(wav)),
            }
        )
        chunks.append(wav)
    sync_device()
    done_s = offset_s + (time.perf_counter() - start_time)
    timeline.append({"time_s": done_s, "source": source, "event": "audio_done", "chunks": chunk_count})
    if not chunks:
        raise RuntimeError(f"{source} stream produced no audio chunks")
    return np.concatenate(chunks), stream_sr, first_audio_s, done_s, chunk_count


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def maybe_write_wav(out_dir: Path, enabled: bool, engine: str, mode: str, kind: str, output_limit: int, audio, sr) -> str:
    if not enabled:
        return ""
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    path = wav_dir / f"{engine}_{mode}_{kind}_{output_limit}.wav"
    sf.write(path, audio, sr)
    return str(path)


def serialize_openai_result(result: OpenAIStreamResult) -> dict:
    data = asdict(result)
    data["deltas"] = [asdict(delta) for delta in result.deltas]
    data["request_tokens_per_second"] = result.request_tokens_per_second
    data["generation_tokens_per_second"] = result.generation_tokens_per_second
    return data


def row_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def warmup_tts_model(args, model, engine: str, mode: str) -> None:
    if args.no_tts_warmup:
        return
    warm_args = Namespace(**vars(args))
    warm_args.max_new_tokens = min(args.max_new_tokens, 2)
    warm_args.min_new_tokens = 1
    warm_args.chunk_size = 1
    warm_text = "Wimbledon changed as tennis modernized."
    try:
        for _ in fulltext_stream(model, engine, mode, warm_text, warm_args):
            break
        sync_device()
    except Exception as exc:
        raise RuntimeError(f"TTS warmup failed for {engine}/{mode}: {exc}") from exc


def run_tts_case(args, output_limit: int, engine: str, mode: str, csv_writer) -> None:
    sync_device()
    model = load_engine(args, engine, mode)
    sync_device()

    prompt = build_wimbledon_prompt(output_limit, args.prompt)
    timeline: list[dict] = []
    print(f"warming TTS: engine={engine}, mode={mode}", flush=True)
    warmup_t0 = time.perf_counter()
    warmup_tts_model(args, model, engine, mode)
    warmup_done_s = time.perf_counter() - warmup_t0
    timeline.append({"time_s": 0.0, "source": f"{engine}:{mode}", "event": "model_hot"})

    print(
        f"requesting live OpenAI stream: model={args.openai_model}, "
        f"max_output_tokens={output_limit}, engine={engine}, mode={mode}"
    )
    live_source, openai_thread, openai_state = start_live_openai_delta_source(args, output_limit, prompt, timeline)
    text_delta_t0 = openai_state["start_time"]
    timeline.append({"time_s": 0.0, "source": f"{engine}:{mode}:text_delta", "event": "tts_start"})
    delta_audio, delta_sr, delta_first, delta_done, delta_chunks = drain_audio_stream(
        text_delta_stream(model, engine, mode, live_source, args),
        timeline,
        f"{engine}:{mode}:text_delta",
        text_delta_t0,
        offset_s=0.0,
    )
    openai_thread.join()
    if openai_state["error"] is not None:
        raise openai_state["error"]
    result = live_state_to_result(args, output_limit, prompt, openai_state, timeline)

    response_stem = f"{engine}_{mode}_{output_limit}"
    write_json(args.out_dir / "openai" / f"{response_stem}.json", serialize_openai_result(result))
    text_path = args.out_dir / "openai" / f"{response_stem}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(result.text, encoding="utf-8")

    full_t0 = time.perf_counter()
    timeline.append(
        {
            "time_s": result.completed_s,
            "source": f"{engine}:{mode}:fulltext",
            "event": "tts_start_after_llm_done",
        }
    )
    full_audio, full_sr, full_first, full_done, full_chunks = drain_audio_stream(
        fulltext_stream(model, engine, mode, result.text, args),
        timeline,
        f"{engine}:{mode}:fulltext",
        full_t0,
        offset_s=result.completed_s,
    )

    text_tokens = count_tokens(model, engine, result.text)
    delta_wav = maybe_write_wav(args.out_dir, args.write_wavs, engine, mode, "text_delta", result.output_limit, delta_audio, delta_sr)
    full_wav = maybe_write_wav(args.out_dir, args.write_wavs, engine, mode, "fulltext", result.output_limit, full_audio, full_sr)

    timeline_path = args.out_dir / "timelines" / f"{response_stem}.jsonl"
    write_jsonl(timeline_path, sorted(timeline, key=lambda item: item.get("time_s", 0.0)))

    llm_done = result.completed_s
    csv_writer.writerow(
        {
            "openai_model": result.model,
            "output_limit": result.output_limit,
            "engine": engine,
            "mode": mode,
            "openai_status": result.status,
            "openai_output_tokens": result.output_tokens,
            "tts_text_tokens": text_tokens,
            "openai_deltas": len(result.deltas),
            "openai_first_token_s": row_value(result.first_token_s),
            "openai_done_s": row_value(result.completed_s),
            "openai_request_tokens_per_s": row_value(result.request_tokens_per_second),
            "openai_generation_tokens_per_s": row_value(result.generation_tokens_per_second),
            "text_delta_first_audio_s": row_value(delta_first),
            "text_delta_audio_done_s": row_value(delta_done),
            "text_delta_audio_before_llm_done_s": row_value(llm_done - delta_first if delta_first is not None else None),
            "fulltext_first_audio_s": row_value(full_first),
            "fulltext_audio_done_s": row_value(full_done),
            "tts_warmup_s": row_value(warmup_done_s),
            "text_delta_chunks": delta_chunks,
            "fulltext_chunks": full_chunks,
            "text_delta_hit_tts_token_cap": delta_chunks * args.chunk_size >= args.max_new_tokens,
            "fulltext_hit_tts_token_cap": full_chunks * args.chunk_size >= args.max_new_tokens,
            "timeline_jsonl": str(timeline_path),
            "text_delta_wav": delta_wav,
            "fulltext_wav": full_wav,
        }
    )
    print(
        f"{engine},{mode},{result.output_limit},"
        f"openai_first={row_value(result.first_token_s)}s,"
        f"openai_done={row_value(result.completed_s)}s,"
        f"tps={row_value(result.generation_tokens_per_second)},"
        f"text_delta_first_audio={row_value(delta_first)}s,"
        f"fulltext_first_audio={row_value(full_first)}s,"
        f"warmup={row_value(warmup_done_s)}s"
    )

    del model
    sync_device()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; export it before running this live benchmark.")
    if "faster" in args.engines and not torch.cuda.is_available():
        raise RuntimeError("faster-qwen3-tts comparison requires CUDA.")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "live_openai_to_live_tts",
        "openai_model": args.openai_model,
        "output_limits": args.output_limits,
        "engines": args.engines,
        "modes": args.modes,
        "device": args.device,
        "chunk_size": args.chunk_size,
        "token_holdback": args.token_holdback,
        "tts_warmup_enabled": not args.no_tts_warmup,
    }
    write_json(args.out_dir / "run_metadata.json", metadata)

    csv_path = args.out_dir / "summary.csv"
    fields = [
        "openai_model",
        "output_limit",
        "engine",
        "mode",
        "openai_status",
        "openai_output_tokens",
        "tts_text_tokens",
        "openai_deltas",
        "openai_first_token_s",
        "openai_done_s",
        "openai_request_tokens_per_s",
        "openai_generation_tokens_per_s",
        "text_delta_first_audio_s",
        "text_delta_audio_done_s",
        "text_delta_audio_before_llm_done_s",
        "fulltext_first_audio_s",
        "fulltext_audio_done_s",
        "tts_warmup_s",
        "text_delta_chunks",
        "fulltext_chunks",
        "text_delta_hit_tts_token_cap",
        "fulltext_hit_tts_token_cap",
        "timeline_jsonl",
        "text_delta_wav",
        "fulltext_wav",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()

        for output_limit in args.output_limits:
            for mode in args.modes:
                for engine in args.engines:
                    run_tts_case(args, output_limit, engine, mode, csv_writer)
                    handle.flush()

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
