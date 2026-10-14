# The WAV reader and chunked PlayStream playback are derived from the G1 audio example
# in Unitree SDK2 Python (https://github.com/unitreerobotics/unitree_sdk2_python),
# Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"),
# BSD 3-Clause. See NOTICES.
import struct
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from lib.deploy.deploy_utils.networks import is_mock_network


def _resolve_wav_path(wav_path: str) -> Path:
    from lib.assets.paths import resolve_asset

    raw = wav_path.strip()
    path = Path(raw)
    if not path.is_absolute():
        return Path(resolve_asset(raw))
    return path.expanduser().resolve()


def _read_wav_file(wav_path: str) -> Tuple[bytes, int, int]:
    path = _resolve_wav_path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path} (resolved: {path})")

    with path.open("rb") as f:
        def read(fmt: str):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

        chunk_id, = read("<I")
        if chunk_id != 0x46464952:  # "RIFF"
            raise ValueError(f"Invalid WAV file: expected RIFF header, got {hex(chunk_id)}")

        _chunk_size, = read("<I")
        format_tag, = read("<I")
        if format_tag != 0x45564157:  # "WAVE"
            raise ValueError(f"Invalid WAV file: expected WAVE format, got {hex(format_tag)}")

        subchunk1_id, = read("<I")
        subchunk1_size, = read("<I")

        if subchunk1_id == 0x4B4E554A:  # "JUNK"
            f.seek(subchunk1_size, 1)
            subchunk1_id, = read("<I")
            subchunk1_size, = read("<I")

        if subchunk1_id != 0x20746D66:  # "fmt "
            raise ValueError(f"Invalid WAV file: expected fmt chunk, got {hex(subchunk1_id)}")
        if subchunk1_size not in (16, 18):
            raise ValueError(f"Unsupported fmt chunk size: {subchunk1_size}")

        audio_format, = read("<H")
        num_channels, = read("<H")
        sample_rate, = read("<I")
        byte_rate, = read("<I")
        block_align, = read("<H")
        bits_per_sample, = read("<H")

        if audio_format != 1:
            raise ValueError(f"Only PCM WAV is supported, got format {audio_format}")
        if bits_per_sample != 16:
            raise ValueError(f"Only 16-bit WAV is supported, got {bits_per_sample} bits")

        expected_byte_rate = sample_rate * num_channels * bits_per_sample // 8
        if byte_rate != expected_byte_rate:
            raise ValueError(
                f"Invalid byte rate in WAV header: got {byte_rate}, expected {expected_byte_rate}"
            )

        expected_align = num_channels * bits_per_sample // 8
        if block_align != expected_align:
            raise ValueError(
                f"Invalid block align in WAV header: got {block_align}, expected {expected_align}"
            )

        if subchunk1_size == 18:
            extra_size, = read("<H")
            if extra_size != 0:
                raise ValueError(f"Unsupported WAV extra data size: {extra_size}")

        while True:
            subchunk2_id, subchunk2_size = read("<II")
            if subchunk2_id == 0x61746164:  # "data"
                break
            f.seek(subchunk2_size, 1)

        pcm_data = f.read(subchunk2_size)
        if len(pcm_data) != subchunk2_size:
            raise ValueError("Failed to read full PCM payload from WAV file")

    return pcm_data, sample_rate, num_channels


def _scale_int16_pcm_mono(pcm_data: bytes, linear_gain: float) -> bytes:
    """Apply linear gain to 16-bit little-endian mono PCM (copies if gain is 1)."""
    if linear_gain == 1.0:
        return pcm_data
    x = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) * linear_gain
    np.clip(x, -32768.0, 32767.0, out=x)
    return x.astype(np.int16).tobytes()


class G1AudioPlayer:
    """Small wrapper around Unitree G1 audio playback.

    Requires the Unitree SDK channel factory to be initialized already
    (e.g. by ``Controller`` / ``G1Env`` before constructing this player).
    """

    def __init__(
        self,
        network: Optional[str] = None,
        *,
        timeout_s: float = 10.0,
        default_stream_name: str = "g1_audio",
        chunk_size: int = 96000,
    ) -> None:
        self.network = network
        self.enabled = not is_mock_network(network)
        self.default_stream_name = default_stream_name
        self.chunk_size = chunk_size
        self._lock = threading.Lock()
        self._play_thread: Optional[threading.Thread] = None
        self._client: Optional[Any] = None

        if not self.enabled:
            return

        try:
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        except ImportError:
            self.enabled = False
            return

        client = AudioClient()
        client.SetTimeout(timeout_s)
        client.Init()
        self._client = client

    def play(
        self,
        wav_path: str,
        stream_name: Optional[str] = None,
        *,
        volume: Optional[float] = None,
    ) -> str:
        stream_name = stream_name or self.default_stream_name
        stream_id = str(int(time.time() * 1000))

        if not self.enabled:
            return stream_id

        gain = 1.0 if volume is None else float(volume)

        self._play_thread = threading.Thread(
            target=self._play_stream,
            args=(wav_path, stream_name, stream_id, gain),
            daemon=True,
        )
        self._play_thread.start()

        return stream_id

    def stop(self, stream_name: Optional[str] = None) -> None:
        if not self.enabled or self._client is None:
            return
        with self._lock:
            self._client.PlayStop(stream_name or self.default_stream_name)

    def _play_stream(
        self,
        wav_path: str,
        stream_name: str,
        stream_id: str,
        linear_gain: float,
    ) -> None:
        if not self.enabled or self._client is None:
            return
        pcm_data, sample_rate, num_channels = _read_wav_file(wav_path)
        if sample_rate != 16000 or num_channels != 1:
            raise ValueError(
                f"Unsupported WAV format for G1 audio: sample_rate={sample_rate}, "
                f"num_channels={num_channels}. Expected 16000 Hz mono."
            )

        pcm_data = _scale_int16_pcm_mono(pcm_data, linear_gain)

        bytes_per_second = sample_rate * num_channels * 2

        with self._lock:
            offset = 0
            while offset < len(pcm_data):
                chunk = pcm_data[offset:offset + self.chunk_size]
                ret_code, _ = self._client.PlayStream(stream_name, stream_id, chunk)
                if ret_code != 0:
                    raise RuntimeError(
                        f"Audio playback failed for stream '{stream_name}', return code {ret_code}"
                    )

                offset += len(chunk)
                if offset < len(pcm_data):
                    time.sleep(len(chunk) / bytes_per_second)
