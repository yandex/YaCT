"""Minimal video compression helpers for MuJoCo env video export."""

import shutil
import subprocess
import tempfile
from typing import List

import numpy as np


def np2mp4bytes(
    frames: List[np.ndarray],
    fps: int = 25,
    threads: int = 8,
    preset: str = "fast",
    codec: str = "libx264",
) -> bytes:
    """Encode RGB uint8 frames to MP4 bytes via ffmpeg (H.264 + yuv420p).

    H.264/yuv420p plays in browsers and VS Code (the Chromium media stack), not
    only in VLC/ffmpeg. OpenCV's ``mp4v`` fourcc writes MPEG-4 Part 2, which
    Chromium cannot decode, so videos looked unplayable despite being valid.
    """
    if not frames:
        raise ValueError("np2mp4bytes requires at least one frame")
    first_frame = frames[0]
    assert first_frame.dtype == np.uint8, first_frame.dtype
    height, width = first_frame.shape[:2]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required to encode video")

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-threads", str(threads),
        # yuv420p needs even dimensions; pad if a frame is odd-sized
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags", "+faststart",
    ]
    with tempfile.NamedTemporaryFile(suffix=".mp4") as out, tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(
            cmd + [out.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=errf,
        )
        try:
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            pass  # ffmpeg exited early; the returncode check below reports why
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            errf.seek(0)
            raise RuntimeError(f"ffmpeg failed: {errf.read().decode(errors='replace')}")
        with open(out.name, "rb") as handle:
            return handle.read()
