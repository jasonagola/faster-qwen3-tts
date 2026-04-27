#!/usr/bin/env python3
"""Compare full-text audio streaming vs text-delta input streaming.

The timeline starts when the simulated upstream LLM begins emitting text.

Rows compare:
  - full-text backend streaming: wait for full LLM text, then start TTS audio streaming
  - text-delta input streaming: feed partial LLM text into TTS as deltas arrive
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf
import torch


THIS_REPO = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_REPO = THIS_REPO.parent / "Qwen3-TTS"
sys.path.insert(0, str(THIS_REPO))
if DEFAULT_UPSTREAM_REPO.exists():
    sys.path.insert(0, str(DEFAULT_UPSTREAM_REPO))

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402
from faster_qwen3_tts.text_delta import split_token_budget_deltas, token_counted_delta_delays  # noqa: E402


DEFAULT_TEXT = (
    "hey, I'm just a little Mac mini who wanted to do the best job he could. "
    "If I had to tell you a story I would start way back when my grandfather was a little boy in Italy."
)
DEFAULT_REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
DEFAULT_REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Qwen3-TTS text-delta input streaming implementations.")
    parser.add_argument("--upstream-repo", type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--engines", nargs="+", choices=["upstream", "faster"], default=["upstream", "faster"])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["custom_voice", "voice_design", "voice_clone_xvec", "voice_clone_icl"],
        default=["custom_voice"],
    )
    parser.add_argument("--custom-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--base-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--voice-design-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--base-text", default=DEFAULT_TEXT)
    parser.add_argument("--multipliers", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--voice-design-instruct", default="Speak clearly in a calm, neutral adult voice.")
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--decode-left-context-frames", type=int, default=25)
    parser.add_argument("--token-holdback", type=int, default=1)
    parser.add_argument("--llm-tokens-per-second", type=float, default=28.0)
    parser.add_argument("--tokens-per-delta", type=int, default=4)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--out-dir", type=Path, default=None)
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


def move_speech_tokenizer_to_device(model, device: str):
    speech_tokenizer = getattr(getattr(model, "model", model), "speech_tokenizer", None)
    if speech_tokenizer is not None and getattr(speech_tokenizer, "model", None) is not None:
        speech_tokenizer.model.to(device)
        speech_tokenizer.device = torch.device(device)


def start_timed_delta_source(delayed_deltas, state: dict):
    sentinel = object()
    pending = queue.Queue()

    def produce():
        for delta, token_count, delay_seconds in delayed_deltas:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            state["emitted_deltas"] += 1
            state["emitted_tokens"] += token_count
            pending.put(delta)
        state["llm_done_seconds"] = time.perf_counter() - state["start_time"]
        pending.put(sentinel)

    def consume():
        while True:
            item = pending.get()
            if item is sentinel:
                break
            yield item

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    return consume(), producer


def stream_to_wav(stream_iter):
    chunks = []
    stream_sr = None
    first_audio_seconds = None
    t0 = time.perf_counter()
    for item in stream_iter:
        wav, stream_sr = item[:2]
        sync_device()
        if first_audio_seconds is None:
            first_audio_seconds = time.perf_counter() - t0
        chunks.append(wav)
    sync_device()
    audio_done_seconds = time.perf_counter() - t0
    if not chunks:
        raise RuntimeError("stream produced no audio chunks")
    return np.concatenate(chunks), stream_sr, first_audio_seconds, audio_done_seconds, len(chunks)


def get_model_id(args, mode: str):
    if mode == "custom_voice":
        return args.custom_model
    if mode == "voice_design":
        return args.voice_design_model
    return args.base_model


def load_engine(args, engine: str, mode: str):
    dtype = resolve_dtype(args.dtype)
    model_id = get_model_id(args, mode)
    if engine == "upstream":
        if str(args.upstream_repo) not in sys.path:
            sys.path.insert(0, str(args.upstream_repo))
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=args.device,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        )
        model.device = torch.device(args.device)
        move_speech_tokenizer_to_device(model, args.device)
        return model

    return FasterQwen3TTS.from_pretrained(
        model_id,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        max_seq_len=args.max_seq_len,
    )


def count_tokens(model, engine: str, text: str) -> int:
    if engine == "upstream":
        return len(model._tokenize_assistant_content(text))
    return len(model._tokenize_assistant_content(text))


def fulltext_stream(model, engine: str, mode: str, text: str, args):
    common = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
    )
    if engine == "upstream":
        stream_common = dict(
            audio_chunk_code_frames=args.chunk_size,
            decode_left_context_frames=args.decode_left_context_frames,
            token_holdback=args.token_holdback,
            **common,
        )
        if mode == "custom_voice":
            return model.stream_custom_voice_from_text_deltas(
                [text], speaker=args.speaker, language=args.language, instruct=args.instruct or None, **stream_common
            )
        if mode == "voice_design":
            return model.stream_voice_design_from_text_deltas(
                [text], instruct=args.voice_design_instruct, language=args.language, **stream_common
            )
        return model.stream_voice_clone_from_text_deltas(
            [text],
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=(mode == "voice_clone_xvec"),
            **stream_common,
        )

    stream_common = dict(chunk_size=args.chunk_size, **common)
    if mode == "custom_voice":
        return model.generate_custom_voice_streaming(
            text, speaker=args.speaker, language=args.language, instruct=args.instruct or None, **stream_common
        )
    if mode == "voice_design":
        return model.generate_voice_design_streaming(
            text, instruct=args.voice_design_instruct, language=args.language, **stream_common
        )
    return model.generate_voice_clone_streaming(
        text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        xvec_only=(mode == "voice_clone_xvec"),
        **stream_common,
    )


def text_delta_stream(model, engine: str, mode: str, text_deltas: Iterable[str], args):
    common = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        token_holdback=args.token_holdback,
    )
    if engine == "upstream":
        stream_common = dict(
            audio_chunk_code_frames=args.chunk_size,
            decode_left_context_frames=args.decode_left_context_frames,
            **common,
        )
        if mode == "custom_voice":
            return model.stream_custom_voice_from_text_deltas(
                text_deltas, speaker=args.speaker, language=args.language, instruct=args.instruct or None, **stream_common
            )
        if mode == "voice_design":
            return model.stream_voice_design_from_text_deltas(
                text_deltas, instruct=args.voice_design_instruct, language=args.language, **stream_common
            )
        return model.stream_voice_clone_from_text_deltas(
            text_deltas,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=(mode == "voice_clone_xvec"),
            **stream_common,
        )

    stream_common = dict(chunk_size=args.chunk_size, **common)
    if mode == "custom_voice":
        return model.stream_custom_voice_from_text_deltas(
            text_deltas, speaker=args.speaker, language=args.language, instruct=args.instruct or None, **stream_common
        )
    if mode == "voice_design":
        return model.stream_voice_design_from_text_deltas(
            text_deltas, instruct=args.voice_design_instruct, language=args.language, **stream_common
        )
    return model.stream_voice_clone_from_text_deltas(
        text_deltas,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        xvec_only=(mode == "voice_clone_xvec"),
        **stream_common,
    )


def maybe_write_wav(args, engine: str, mode: str, kind: str, multiplier: int, audio, sr) -> str:
    if args.out_dir is None:
        return ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{engine}_{mode}_{kind}_{multiplier}x.wav"
    sf.write(path, audio, sr)
    return str(path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for faster-qwen3-tts comparison.")

    print(
        "engine,mode,multiplier,text_tokens,deltas,scheduled_llm_done_s,"
        "fulltext_first_audio_s,fulltext_audio_done_s,"
        "text_delta_first_audio_s,text_delta_llm_done_s,text_delta_audio_done_s,"
        "text_delta_audio_before_llm_done_s,text_delta_source_exhausted,"
        "fulltext_chunks,text_delta_chunks,fulltext_wav,text_delta_wav"
    )

    for mode in args.modes:
        for engine in args.engines:
            sync_device()
            model = load_engine(args, engine, mode)
            sync_device()

            for multiplier in args.multipliers:
                text = " ".join([args.base_text] * multiplier)
                text_tokens = count_tokens(model, engine, text)
                deltas = split_token_budget_deltas(
                    text,
                    args.tokens_per_delta,
                    lambda s: count_tokens(model, engine, s),
                )
                delayed_deltas = token_counted_delta_delays(
                    deltas,
                    args.llm_tokens_per_second,
                    lambda s: count_tokens(model, engine, s),
                )
                scheduled_llm_done = sum(delay for _, _, delay in delayed_deltas)

                full_audio, full_sr, full_first, full_done, full_chunks = stream_to_wav(
                    fulltext_stream(model, engine, mode, text, args)
                )
                full_first_timeline = scheduled_llm_done + full_first
                full_done_timeline = scheduled_llm_done + full_done
                full_wav = maybe_write_wav(args, engine, mode, "fulltext", multiplier, full_audio, full_sr)

                state = {
                    "start_time": None,
                    "emitted_deltas": 0,
                    "emitted_tokens": 0,
                    "llm_done_seconds": None,
                }
                sync_device()
                state["start_time"] = time.perf_counter()
                delta_source, producer = start_timed_delta_source(delayed_deltas, state)
                delta_audio, delta_sr, delta_first, _, delta_chunks = stream_to_wav(
                    text_delta_stream(model, engine, mode, delta_source, args)
                )
                sync_device()
                delta_done = time.perf_counter() - state["start_time"]
                producer.join()
                delta_wav = maybe_write_wav(args, engine, mode, "text_delta", multiplier, delta_audio, delta_sr)

                llm_done = state["llm_done_seconds"]
                audio_before_llm_done = llm_done - delta_first if llm_done is not None else float("nan")
                source_exhausted = state["emitted_deltas"] == len(deltas)

                print(
                    f"{engine},"
                    f"{mode},"
                    f"{multiplier},"
                    f"{text_tokens},"
                    f"{len(deltas)},"
                    f"{scheduled_llm_done:.2f},"
                    f"{full_first_timeline:.2f},"
                    f"{full_done_timeline:.2f},"
                    f"{delta_first:.2f},"
                    f"{llm_done:.2f},"
                    f"{delta_done:.2f},"
                    f"{audio_before_llm_done:.2f},"
                    f"{source_exhausted},"
                    f"{full_chunks},"
                    f"{delta_chunks},"
                    f"{full_wav},"
                    f"{delta_wav}"
                )

            del model
            sync_device()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
