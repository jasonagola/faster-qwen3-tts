#!/usr/bin/env python3
"""Four-way LLM-to-TTS benchmark normalized to the first prepared token.

The benchmark uses prepared local text and replays exactly N text deltas at a
fixed token rate. This removes live LLM output variance while preserving the
timeline shape of an LLM streaming text into TTS.

Default comparison:

1. Vanilla Qwen3-TTS: full text in, complete audio out.
2. Qwen3-TTS-streaming: full text in, first streamed audio chunk out.
3. FasterQwen3TTS: full text in, first streamed audio chunk out.
4. This fork: text deltas in, first streamed audio chunk out.

The `qwen_tts` package name collides between vanilla and streaming forks, so
those measurements run in isolated subprocesses with explicit PYTHONPATHs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


THIS_REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CODEC_SAMPLES_PER_FRAME = 1920
DEFAULT_REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up "
    "reinforcement learning atop LLMs."
)
PREPARED_TEXT_SOURCE = """
Wimbledon has always balanced ritual with reinvention. The white clothing, the
clipped grass, and the careful silence before a serve still make the tournament
feel tied to another age, yet the tennis itself has changed dramatically. Wooden
rackets rewarded touch, timing, and quick approaches to the net. Graphite
rackets brought more power, more spin, and a baseline game that can stretch a
rally from corner to corner. Serve and volley once defined grass court instinct,
but modern players defend, slide, recover, and counter with athletic patterns
shaped by sports science.

The courts changed too, not by abandoning grass, but by making it more durable,
more even, and more compatible with longer rallies. The old surface could be
slick and uneven by the second week, with bad bounces that pushed players
toward quick points. Modern maintenance has made the bounce truer and the game
less chaotic. A player can still attack, but the reward now comes from choosing
the right moment rather than charging forward because the surface demands it.

Equipment changed the sound and rhythm of the tournament. A wooden racket made
every strike feel like a negotiation. The sweet spot was small, and a mistimed
swing could turn courage into embarrassment. Graphite widened the margin for
violence. Players could swing harder, brush the ball with heavier topspin, and
still land it near the baseline. That technology did not remove skill; it moved
skill into new places, toward acceleration, recovery, disguise, and the ability
to repeat explosive movement for hours.

The athletes changed with the tools. Training teams became larger and more
specialized. Fitness coaches, physiotherapists, nutritionists, analysts, and
stringers now shape the performance that appears on court. The best players are
not only shot makers; they are systems for solving pressure. They prepare for
patterns, study tendencies, manage hydration, and adjust rackets for weather
that the crowd may barely notice. The old mythology of tennis as a solitary
duel still matters, but the modern champion stands at the visible edge of a
large invisible operation.

Technology changed the argument around fairness. Hawk Eye did not eliminate
disagreement, but it changed the emotional contract between player, official,
and spectator. The old line-call drama had a human texture, and sometimes a
human injustice. Review systems made the game feel more precise, more
accountable, and less dependent on one instant of eyesight. Roofed courts
changed another old tension. Rain delays once bent the shape of entire draws.
Now the tournament can preserve its schedule while still carrying the mood of
English weather just outside the lights.

Money and media changed the meaning of Wimbledon beyond the grounds. Prize
money made professional tennis more viable for more players, while global
coverage turned a local ritual into a worldwide appointment. A point played on
Centre Court can be clipped, argued about, and studied everywhere before the
players have left their chairs. That visibility raises pressure, but it also
preserves moments that would once have lived only in memory.

For all of that change, Wimbledon still works because it lets the old and new
press against each other. The tournament looks backward through its ceremonies
and forward through the speed of the ball. It asks modern athletes to perform
inside an old frame, and that tension gives the event its force. The grass is
still grass. The silence before serve is still silence. But the problem placed
before each player is faster, stronger, more technical, and more public than it
used to be.
"""


@dataclass
class TextRecording:
    target_tokens: int
    text: str
    done_after_first_token_s: float
    delta_events: list[dict]


@dataclass
class Measurement:
    label: str
    first_audio_s: Optional[float]
    audio_done_s: Optional[float]
    chunks: int = 0
    samples: int = 0
    sample_rate: Optional[int] = None
    hit_cap: Optional[bool] = None
    note: str = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Four-way prepared-token LLM-to-TTS benchmark.")
    parser.add_argument("--targets", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--simulated-tokens-per-second", type=float, default=30.0)
    parser.add_argument(
        "--delta-tokenizer",
        choices=["openai-o200k", "qwen", "whitespace"],
        default="openai-o200k",
        help="Tokenizer used to convert prepared text into replayed text deltas.",
    )
    parser.add_argument(
        "--delta-tokenizer-model",
        default=None,
        help="Tokenizer model for --delta-tokenizer qwen. Defaults to --model.",
    )
    parser.add_argument(
        "--source-text-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file to slice into token prefixes. Defaults to the built-in Wimbledon text.",
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--language", default="English")
    parser.add_argument("--ref-audio", default="ref_audio.wav")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--token-holdback", type=int, default=1)
    parser.add_argument(
        "--text-delta-token-holdbacks",
        nargs="+",
        type=int,
        default=None,
        help="One or more token_holdback values to benchmark for the text-delta path. Defaults to --token-holdback.",
    )
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--do-sample", dest="do_sample", action="store_true")
    sampling.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=False)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--out-dir", type=Path, default=Path("text_delta_normalized_benchmark"))
    parser.add_argument("--vanilla-repo", type=Path, default=Path("../Qwen3-TTS-vanilla"))
    parser.add_argument("--streaming-repo", type=Path, default=Path("../Qwen3-TTS-streaming"))
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--skip-qwen-streaming", action="store_true")
    parser.add_argument("--write-wavs", action="store_true")

    parser.add_argument("--worker-kind", choices=["vanilla", "qwen_streaming"], default=None)
    parser.add_argument("--worker-config", type=Path, default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
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


def whitespace_prepared_deltas(target_tokens: int, source_text: str) -> list[str]:
    source_words = source_text.split()
    if len(source_words) < target_tokens:
        raise RuntimeError(
            f"The source text has only {len(source_words)} whitespace tokens; "
            f"cannot prepare a {target_tokens}-token prefix."
        )
    words = source_words[:target_tokens]
    return [word + (" " if index < len(words) - 1 else "") for index, word in enumerate(words)]


def token_ids_for_text(text: str, tokenizer_kind: str, tokenizer_model: str) -> tuple[list[int], object]:
    if tokenizer_kind == "openai-o200k":
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "The openai-o200k delta tokenizer requires tiktoken. "
                "Install it with `python -m pip install tiktoken`, or pass `--delta-tokenizer qwen`."
            ) from exc
        tokenizer = tiktoken.get_encoding("o200k_base")
        return tokenizer.encode(text), tokenizer

    if tokenizer_kind == "qwen":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        return tokenizer.encode(text, add_special_tokens=False), tokenizer

    raise ValueError(f"Token ids are not available for tokenizer kind: {tokenizer_kind}")


def decode_tokens(tokenizer, token_ids: list[int], tokenizer_kind: str) -> str:
    if tokenizer_kind == "openai-o200k":
        return tokenizer.decode(token_ids)
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def tokenized_prepared_deltas(
    target_tokens: int,
    source_text: str,
    tokenizer_kind: str,
    tokenizer_model: str,
) -> list[str]:
    if tokenizer_kind == "whitespace":
        return whitespace_prepared_deltas(target_tokens, source_text)

    token_ids, tokenizer = token_ids_for_text(source_text, tokenizer_kind, tokenizer_model)
    if len(token_ids) < target_tokens:
        raise RuntimeError(
            f"The source text has only {len(token_ids)} {tokenizer_kind} tokens; "
            f"cannot prepare a {target_tokens}-token prefix."
        )

    selected = token_ids[:target_tokens]
    deltas: list[str] = []
    previous = ""
    for index in range(len(selected)):
        prefix = decode_tokens(tokenizer, selected[: index + 1], tokenizer_kind)
        delta = prefix[len(previous):]
        deltas.append(delta)
        previous = prefix
    return deltas


def build_recordings(
    targets: list[int],
    tokens_per_second: float,
    source_text: str,
    tokenizer_kind: str,
    tokenizer_model: str,
) -> list[TextRecording]:
    if tokens_per_second <= 0:
        raise ValueError("--simulated-tokens-per-second must be > 0.")

    recordings = []
    for target in targets:
        deltas = tokenized_prepared_deltas(target, source_text, tokenizer_kind, tokenizer_model)
        delta_events = [
            {"time_s": index / tokens_per_second, "delta": delta}
            for index, delta in enumerate(deltas)
        ]
        recordings.append(
            TextRecording(
                target_tokens=target,
                text="".join(deltas).strip(),
                done_after_first_token_s=delta_events[-1]["time_s"] if delta_events else 0.0,
                delta_events=delta_events,
            )
        )
    return recordings


def make_delta_replay(recording: TextRecording):
    start = time.perf_counter()

    def replay():
        for event in recording.delta_events:
            sleep_s = start + float(event["time_s"]) - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            yield event["delta"]

    return replay(), start


def generation_kwargs(config: dict, include_sampling: bool = True) -> dict:
    kwargs = {
        "max_new_tokens": config["max_new_tokens"],
        "min_new_tokens": config["min_new_tokens"],
        "do_sample": config["do_sample"],
        "repetition_penalty": config["repetition_penalty"],
    }
    if config["do_sample"] and include_sampling:
        kwargs.update(
            {
                "temperature": config["temperature"],
                "top_k": config["top_k"],
                "top_p": config["top_p"],
            }
        )
    return kwargs


def qwen_streaming_generation_kwargs(config: dict) -> dict:
    kwargs = generation_kwargs(config)
    kwargs.pop("repetition_penalty", None)
    return kwargs


def normalize_audio(audio) -> np.ndarray:
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio).reshape(-1)


def drain_stream(stream_iter, label: str, start_time: float, max_new_tokens: int, chunk_size: int):
    chunks = []
    sample_rate = None
    first_audio_s = None
    for item in stream_iter:
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
    return audio, Measurement(
        label=label,
        first_audio_s=first_audio_s,
        audio_done_s=done_s,
        chunks=len(chunks),
        samples=int(audio.shape[0]),
        sample_rate=int(sample_rate),
        hit_cap=len(chunks) * chunk_size >= max_new_tokens,
    )


def write_wav(config: dict, target: int, label: str, audio: Optional[np.ndarray], sample_rate: Optional[int]):
    if not config["write_wavs"] or audio is None or sample_rate is None:
        return
    path = Path(config["out_dir"]) / "wav" / f"voice_clone_{target}_{label}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def run_faster(config: dict, recordings: list[TextRecording]) -> dict[str, dict[int, Measurement]]:
    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(
        config["model"],
        device=config["device"],
        dtype=resolve_dtype(config["dtype"]),
        attn_implementation=config["attn_implementation"],
        max_seq_len=config["max_seq_len"],
    )

    warm_kwargs = dict(config)
    warm_kwargs["max_new_tokens"] = 24
    warm_kwargs["min_new_tokens"] = 1
    warm_kwargs["chunk_size"] = 1
    warm_stream = model.generate_voice_clone_streaming(
        text="Wimbledon changed as tennis modernized.",
        language=config["language"],
        ref_audio=config["ref_audio"],
        ref_text=config["ref_text"],
        xvec_only=True,
        chunk_size=1,
        **generation_kwargs(warm_kwargs),
    )
    for _ in warm_stream:
        break
    sync_device()

    holdbacks = config["text_delta_token_holdbacks"]
    results: dict[str, dict[int, Measurement]] = {"faster_fulltext": {}}
    for holdback in holdbacks:
        results[f"faster_text_delta_hb{holdback}"] = {}

    for recording in recordings:
        for holdback in holdbacks:
            label = f"faster_text_delta_hb{holdback}"
            deltas, start = make_delta_replay(recording)
            stream = model.stream_voice_clone_from_text_deltas(
                text_deltas=deltas,
                language=config["language"],
                ref_audio=config["ref_audio"],
                ref_text=config["ref_text"],
                xvec_only=True,
                chunk_size=config["chunk_size"],
                token_holdback=holdback,
                **generation_kwargs(config),
            )
            audio, measurement = drain_stream(
                stream, label, start, config["max_new_tokens"], config["chunk_size"]
            )
            results[label][recording.target_tokens] = measurement
            write_wav(config, recording.target_tokens, measurement.label, audio, measurement.sample_rate)

        start = time.perf_counter()
        stream = model.generate_voice_clone_streaming(
            text=recording.text,
            language=config["language"],
            ref_audio=config["ref_audio"],
            ref_text=config["ref_text"],
            xvec_only=True,
            chunk_size=config["chunk_size"],
            **generation_kwargs(config),
        )
        audio, measurement = drain_stream(
            stream, "faster_fulltext", start, config["max_new_tokens"], config["chunk_size"]
        )
        measurement.first_audio_s += recording.done_after_first_token_s
        measurement.audio_done_s += recording.done_after_first_token_s
        results["faster_fulltext"][recording.target_tokens] = measurement
        write_wav(config, recording.target_tokens, measurement.label, audio, measurement.sample_rate)

    del model
    sync_device()
    torch.cuda.empty_cache()
    return results


def qwen_model(config: dict):
    repo = Path(config["repo"]).resolve()
    sys.path.insert(0, str(repo))
    for name in list(sys.modules):
        if name == "qwen_tts" or name.startswith("qwen_tts."):
            sys.modules.pop(name, None)
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        config["model"],
        device_map=config["device"],
        dtype=resolve_dtype(config["dtype"]),
        attn_implementation=config["attn_implementation"],
    )


def worker_vanilla(config: dict, recordings: list[TextRecording]) -> dict[int, Measurement]:
    model = qwen_model(config)
    warm_kwargs = dict(config)
    warm_kwargs["max_new_tokens"] = 24
    warm_kwargs["min_new_tokens"] = 1
    _ = model.generate_voice_clone(
        text="Wimbledon changed as tennis modernized.",
        language=config["language"],
        ref_audio=config["ref_audio"],
        ref_text=config["ref_text"],
        x_vector_only_mode=True,
        **generation_kwargs(warm_kwargs),
    )
    sync_device()

    results = {}
    for recording in recordings:
        start = time.perf_counter()
        audio_list, sample_rate = model.generate_voice_clone(
            text=recording.text,
            language=config["language"],
            ref_audio=config["ref_audio"],
            ref_text=config["ref_text"],
            x_vector_only_mode=True,
            **generation_kwargs(config),
        )
        sync_device()
        elapsed = time.perf_counter() - start
        audio = normalize_audio(audio_list)
        ready_s = recording.done_after_first_token_s + elapsed
        measurement = Measurement(
            label="vanilla_fulltext",
            first_audio_s=ready_s,
            audio_done_s=ready_s,
            chunks=1,
            samples=int(audio.shape[0]),
            sample_rate=int(sample_rate),
            hit_cap=int(audio.shape[0]) >= (config["max_new_tokens"] - 2) * CODEC_SAMPLES_PER_FRAME,
            note="Vanilla Qwen3-TTS returns complete audio; first_audio_s is audio-ready time.",
        )
        results[recording.target_tokens] = measurement
        write_wav(config, recording.target_tokens, measurement.label, audio, measurement.sample_rate)
    return results


def worker_qwen_streaming(config: dict, recordings: list[TextRecording]) -> dict[int, Measurement]:
    model = qwen_model(config)

    if hasattr(model, "enable_streaming_optimizations") and config.get("qwen_streaming_optimized", False):
        model.enable_streaming_optimizations(
            decode_window_frames=config["streaming_decode_window_frames"],
            use_compile=True,
            compile_mode="reduce-overhead",
        )

    warm_kwargs = dict(config)
    warm_kwargs["max_new_tokens"] = 24
    warm_kwargs["min_new_tokens"] = 1
    stream = model.stream_generate_voice_clone(
        text="Wimbledon changed as tennis modernized.",
        language=config["language"],
        ref_audio=config["ref_audio"],
        ref_text=config["ref_text"],
        x_vector_only_mode=True,
        emit_every_frames=1,
        decode_window_frames=config["streaming_decode_window_frames"],
        overlap_samples=0,
        max_frames=24,
        use_optimized_decode=config.get("qwen_streaming_optimized", False),
        first_chunk_emit_every=0,
        repetition_penalty=config["streaming_repetition_penalty"],
        **qwen_streaming_generation_kwargs(warm_kwargs),
    )
    for _ in stream:
        break
    sync_device()

    results = {}
    for recording in recordings:
        start = time.perf_counter()
        stream = model.stream_generate_voice_clone(
            text=recording.text,
            language=config["language"],
            ref_audio=config["ref_audio"],
            ref_text=config["ref_text"],
            x_vector_only_mode=True,
            emit_every_frames=config["chunk_size"],
            decode_window_frames=config["streaming_decode_window_frames"],
            overlap_samples=0,
            max_frames=config["max_new_tokens"],
            use_optimized_decode=config.get("qwen_streaming_optimized", False),
            first_chunk_emit_every=0,
            repetition_penalty=config["streaming_repetition_penalty"],
            **qwen_streaming_generation_kwargs(config),
        )
        audio, measurement = drain_stream(
            stream, "qwen_streaming_fulltext", start, config["max_new_tokens"], config["chunk_size"]
        )
        measurement.first_audio_s += recording.done_after_first_token_s
        measurement.audio_done_s += recording.done_after_first_token_s
        results[recording.target_tokens] = measurement
        write_wav(config, recording.target_tokens, measurement.label, audio, measurement.sample_rate)
    return results


def measurement_to_dict(measurement: Measurement) -> dict:
    return {
        "label": measurement.label,
        "first_audio_s": measurement.first_audio_s,
        "audio_done_s": measurement.audio_done_s,
        "chunks": measurement.chunks,
        "samples": measurement.samples,
        "sample_rate": measurement.sample_rate,
        "hit_cap": measurement.hit_cap,
        "note": measurement.note,
    }


def measurement_from_dict(data: dict) -> Measurement:
    return Measurement(
        label=data["label"],
        first_audio_s=data.get("first_audio_s"),
        audio_done_s=data.get("audio_done_s"),
        chunks=data.get("chunks", 0),
        samples=data.get("samples", 0),
        sample_rate=data.get("sample_rate"),
        hit_cap=data.get("hit_cap"),
        note=data.get("note", ""),
    )


def recordings_to_json(recordings: list[TextRecording]) -> list[dict]:
    return [
        {
            "target_tokens": item.target_tokens,
            "text": item.text,
            "done_after_first_token_s": item.done_after_first_token_s,
            "delta_events": item.delta_events,
        }
        for item in recordings
    ]


def recordings_from_json(data: list[dict]) -> list[TextRecording]:
    return [
        TextRecording(
            target_tokens=item["target_tokens"],
            text=item["text"],
            done_after_first_token_s=item["done_after_first_token_s"],
            delta_events=item["delta_events"],
        )
        for item in data
    ]


def run_worker(kind: str, config: dict, recordings: list[TextRecording], repo: Path) -> dict[int, Measurement]:
    worker_dir = Path(config["out_dir"]) / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    config_path = worker_dir / f"{kind}_config.json"
    output_path = worker_dir / f"{kind}_output.json"
    worker_config = dict(config)
    worker_config["repo"] = str(repo.resolve())
    worker_config["recordings"] = recordings_to_json(recordings)
    config_path.write_text(json.dumps(worker_config, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo.resolve())
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-kind",
        kind,
        "--worker-config",
        str(config_path),
        "--worker-output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, env=env)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    return {int(target): measurement_from_dict(value) for target, value in data.items()}


def worker_main(args) -> None:
    if args.worker_config is None or args.worker_output is None:
        raise ValueError("--worker-config and --worker-output are required for worker mode.")
    config = json.loads(args.worker_config.read_text(encoding="utf-8"))
    recordings = recordings_from_json(config["recordings"])
    if args.worker_kind == "vanilla":
        results = worker_vanilla(config, recordings)
    elif args.worker_kind == "qwen_streaming":
        results = worker_qwen_streaming(config, recordings)
    else:
        raise ValueError(f"Unknown worker kind: {args.worker_kind}")
    payload = {target: measurement_to_dict(value) for target, value in results.items()}
    args.worker_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def saved(a: Optional[Measurement], b: Optional[Measurement]) -> Optional[float]:
    if a is None or b is None or a.first_audio_s is None or b.first_audio_s is None:
        return None
    return a.first_audio_s - b.first_audio_s


def text_delta_result_key(holdback: int) -> str:
    return f"faster_text_delta_hb{holdback}"


def build_rows(
    recordings: list[TextRecording],
    results: dict[str, dict[int, Measurement]],
    holdbacks: list[int],
) -> list[dict]:
    rows = []
    for recording in recordings:
        target = recording.target_tokens
        faster_full = results["faster_fulltext"].get(target)
        qwen_streaming = results.get("qwen_streaming_fulltext", {}).get(target)
        vanilla = results.get("vanilla_fulltext", {}).get(target)
        row = {
            "target_tokens": target,
            "prepared_text_deltas": len(recording.delta_events),
            "prepared_text_chars": len(recording.text),
            "prepared_text_words": len(recording.text.split()),
            "llm_done_after_first_token_s": metric(recording.done_after_first_token_s),
            "faster_fulltext_first_audio_s": metric(faster_full.first_audio_s if faster_full else None),
            "qwen_streaming_first_audio_s": metric(qwen_streaming.first_audio_s if qwen_streaming else None),
            "vanilla_audio_ready_s": metric(vanilla.first_audio_s if vanilla else None),
            "qwen_streaming_gain_vs_vanilla_s": metric(saved(vanilla, qwen_streaming)),
            "faster_gain_vs_qwen_streaming_s": metric(saved(qwen_streaming, faster_full)),
            "qwen_streaming_hit_tts_token_cap": qwen_streaming.hit_cap if qwen_streaming else "",
            "faster_fulltext_hit_tts_token_cap": faster_full.hit_cap if faster_full else "",
            "vanilla_hit_tts_token_cap": vanilla.hit_cap if vanilla else "",
        }
        for holdback in holdbacks:
            faster_delta = results[text_delta_result_key(holdback)].get(target)
            row[f"text_delta_hb{holdback}_first_audio_s"] = metric(
                faster_delta.first_audio_s if faster_delta else None
            )
            row[f"text_delta_hb{holdback}_frontside_gain_vs_faster_fulltext_s"] = metric(
                saved(faster_full, faster_delta)
            )
            row[f"text_delta_hb{holdback}_total_gain_vs_vanilla_s"] = metric(saved(vanilla, faster_delta))
            row[f"text_delta_hb{holdback}_hit_tts_token_cap"] = faster_delta.hit_cap if faster_delta else ""
        rows.append(row)
    return rows


def _float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def holdbacks_from_rows(rows: list[dict]) -> list[int]:
    holdbacks: set[int] = set()
    prefix = "text_delta_hb"
    suffix = "_first_audio_s"
    for row in rows:
        for key in row:
            if key.startswith(prefix) and key.endswith(suffix):
                holdbacks.add(int(key[len(prefix):-len(suffix)]))
    return sorted(holdbacks)


def markdown_tables(rows: list[dict]) -> str:
    holdbacks = holdbacks_from_rows(rows)
    text_delta_headers = [
        f"This repo: Faster + text-delta hb={holdback} first audio"
        for holdback in holdbacks
    ]
    lines = [
        "### Normalized Timeline",
        "",
        "`T+0.000s` is the first prepared stream token. Columns are ordered from fastest expected first audio to slowest baseline.",
        "",
        "| Target | " + " | ".join(text_delta_headers + [
            "LLM done",
            "Faster full-text first audio",
            "Qwen3-TTS-streaming first audio",
            "Vanilla Qwen full-text audio ready",
        ]) + " |",
        "|---:|" + "---:|" * (len(text_delta_headers) + 4),
    ]
    for row in rows:
        text_delta_values = [
            tplus(_float(row.get(f"text_delta_hb{holdback}_first_audio_s")))
            for holdback in holdbacks
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['target_tokens']} tokens",
                    *text_delta_values,
                    tplus(_float(row.get("llm_done_after_first_token_s"))),
                    tplus(_float(row.get("faster_fulltext_first_audio_s"))),
                    tplus(_float(row.get("qwen_streaming_first_audio_s"))),
                    tplus(_float(row.get("vanilla_audio_ready_s"))),
                ]
            )
            + " |"
        )

    frontside_headers = [
        f"Front-side gain hb={holdback} vs Faster full-text"
        for holdback in holdbacks
    ]
    total_headers = [
        f"Total gain hb={holdback} vs vanilla"
        for holdback in holdbacks
    ]
    lines.extend(
        [
            "",
            "### Improvement Breakdown",
            "",
            "🟩 means at least 1s saved, 🟨 means 0.25-1s saved, and ⬜ means under 0.25s saved.",
            "",
            "| Target | Backend audio streaming: Qwen streaming vs vanilla | CUDA graph gain: Faster vs Qwen streaming | "
            + " | ".join(frontside_headers + total_headers)
            + " |",
            "|---:|---:|---:|" + "---:|" * (len(frontside_headers) + len(total_headers)),
        ]
    )
    for row in rows:
        vanilla = _float(row.get("vanilla_audio_ready_s"))
        frontside_values = [
            improvement(_float(row.get(f"text_delta_hb{holdback}_frontside_gain_vs_faster_fulltext_s")))
            for holdback in holdbacks
        ]
        total_values = []
        for holdback in holdbacks:
            total = _float(row.get(f"text_delta_hb{holdback}_total_gain_vs_vanilla_s"))
            this_repo = _float(row.get(f"text_delta_hb{holdback}_first_audio_s"))
            total_values.append(f"{improvement(total)} ({ratio(this_repo, vanilla)})")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['target_tokens']} tokens",
                    improvement(_float(row.get("qwen_streaming_gain_vs_vanilla_s"))),
                    improvement(_float(row.get("faster_gain_vs_qwen_streaming_s"))),
                    *frontside_values,
                    *total_values,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.worker_kind is not None:
        worker_main(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_text = (
        args.source_text_file.read_text(encoding="utf-8")
        if args.source_text_file is not None
        else PREPARED_TEXT_SOURCE.strip()
    )
    delta_tokenizer_model = args.delta_tokenizer_model or args.model
    holdbacks = list(dict.fromkeys(args.text_delta_token_holdbacks or [args.token_holdback]))
    if any(value < 0 for value in holdbacks):
        raise ValueError("Token holdback values must be >= 0.")
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "targets": args.targets,
        "simulated_tokens_per_second": args.simulated_tokens_per_second,
        "delta_tokenizer": args.delta_tokenizer,
        "delta_tokenizer_model": delta_tokenizer_model,
        "source_text_file": str(args.source_text_file.resolve()) if args.source_text_file else None,
        "source_text_chars": len(source_text),
        "language": args.language,
        "ref_audio": str(Path(args.ref_audio).resolve()),
        "ref_text": args.ref_text,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "chunk_size": args.chunk_size,
        "token_holdback": args.token_holdback,
        "text_delta_token_holdbacks": holdbacks,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "streaming_decode_window_frames": 80,
        "streaming_repetition_penalty": 1.0,
        "qwen_streaming_optimized": False,
        "out_dir": str(args.out_dir),
        "write_wavs": args.write_wavs,
        "normalization": "All downstream TTS times are seconds after the first prepared tokenizer delta.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    recordings = build_recordings(
        args.targets,
        args.simulated_tokens_per_second,
        source_text,
        args.delta_tokenizer,
        delta_tokenizer_model,
    )
    recordings_dir = args.out_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    for recording in recordings:
        (recordings_dir / f"prepared_{recording.target_tokens}.json").write_text(
            json.dumps(recordings_to_json([recording])[0], indent=2) + "\n",
            encoding="utf-8",
        )

    results = run_faster(config, recordings)
    if not args.skip_qwen_streaming:
        results["qwen_streaming_fulltext"] = run_worker(
            "qwen_streaming", config, recordings, args.streaming_repo
        )
    if not args.skip_vanilla:
        results["vanilla_fulltext"] = run_worker("vanilla", config, recordings, args.vanilla_repo)

    rows = build_rows(recordings, results, holdbacks)
    fieldnames = sorted({key for row in rows for key in row})
    with (args.out_dir / "summary_normalized.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table = markdown_tables(rows)
    (args.out_dir / "readme_tables.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote four-way benchmark output to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
