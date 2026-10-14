#!/usr/bin/env python3
"""Generate the spoken alerts under assets/tts/alerts/ with Kokoro TTS.

Kokoro-82M (Apache-2.0 weights, https://huggingface.co/hexgrad/Kokoro-82M) speaks at
24 kHz; clips are resampled to the 16 kHz mono 16-bit PCM that the G1 speaker plays.
Requires `pip install kokoro soundfile`. Run from the repository root:

    python3 scripts/generate_alert_audio.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ALERTS = {
    "motor_temp_warning": "Warning. High motor temperatures. Cool down recommended.",
}
VOICE = "af_heart"
SAMPLE_RATE = 16000  # G1 speaker format
TARGET_RMS = 0.11  # loudness the configured alert volumes were tuned for
MAX_PEAK = 0.95
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "tts" / "alerts"


def synthesize(text: str) -> np.ndarray:
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")  # American English
    audio = np.concatenate([np.asarray(chunk) for _, _, chunk in pipeline(text, voice=VOICE)])
    audio = resample_poly(audio, SAMPLE_RATE, 24000)
    gain = min(TARGET_RMS / np.sqrt(np.mean(audio**2)), MAX_PEAK / np.abs(audio).max())
    return audio * gain


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in ALERTS.items():
        pcm = (synthesize(text) * 32767).astype(np.int16)
        path = OUTPUT_DIR / f"{name}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(pcm.tobytes())
        print(f"wrote {path} ({len(pcm) / SAMPLE_RATE:.2f} s): {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
