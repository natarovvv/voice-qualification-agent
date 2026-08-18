"""Voice activity detection + turn gating.

Silero VAD when it is installed, adaptive-energy otherwise, so the pipeline
runs on a machine with no torch. Both expose the same frame -> bool contract.
"""
from __future__ import annotations

import array
import logging
import math

from config import SAMPLE_RATE

log = logging.getLogger(__name__)

WINDOW = 512  # samples; Silero v5 accepts exactly this at 16 kHz

class _Silero:
    def __init__(self, threshold: float) -> None:
        import torch  # noqa: F401  (imported for the tensor bridge below)
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        self.threshold = threshold

    def prob(self, samples: array.array) -> float:
        t = self._torch.tensor(samples, dtype=self._torch.float32) / 32768.0
        with self._torch.no_grad():
            return float(self._model(t, SAMPLE_RATE).item())

    def reset(self) -> None:
        self._model.reset_states()

class _Energy:
    """Adaptive RMS gate. Tracks the noise floor so it survives a hissy mic."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.floor = 200.0

    def prob(self, samples: array.array) -> float:
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) or 1.0
        ratio = rms / max(self.floor, 50.0)
        if ratio < 2.0:  # only adapt on what looks like silence
            self.floor = 0.95 * self.floor + 0.05 * rms
        # map 2x floor -> ~0.5, 6x floor -> ~0.9
        return max(0.0, min(1.0, (ratio - 1.0) / 5.0))

    def reset(self) -> None:
        self.floor = 200.0

def _make_detector(threshold: float):
    try:
        det = _Silero(threshold)
        log.info("VAD: silero")
        return det
    except Exception as exc:  # noqa: BLE001 - torch/silero absent is expected
        log.warning("VAD: falling back to energy gate (%s)", exc)
        return _Energy(threshold)

class SpeechGate:
    """Turns a byte stream of PCM16 into 'start' / 'end' speech events.

    ``update`` accepts any chunk size; it buffers to WINDOW internally.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        start_ms: int = 120,
        end_ms: int = 500,
    ) -> None:
        self.det = _make_detector(threshold)
        self.frame_ms = WINDOW * 1000 // SAMPLE_RATE  # 32 ms
        self.start_frames = max(1, start_ms // self.frame_ms)
        self.end_frames = max(1, end_ms // self.frame_ms)
        self._buf = bytearray()
        self._speech = 0
        self._silence = 0
        self.active = False

    def update(self, pcm: bytes) -> list[str]:
        """Feed audio, get zero or more of 'start' / 'end' in order."""
        events: list[str] = []
        self._buf.extend(pcm)
        step = WINDOW * 2
        while len(self._buf) >= step:
            window = array.array("h", bytes(self._buf[:step]))
            del self._buf[:step]
            voiced = self.det.prob(window) >= self.det.threshold
            if voiced:
                self._speech += 1
                self._silence = 0
            else:
                self._silence += 1
                self._speech = 0
            if not self.active and self._speech >= self.start_frames:
                self.active = True
                events.append("start")
            elif self.active and self._silence >= self.end_frames:
                self.active = False
                events.append("end")
        return events

    def reset(self) -> None:
        self._buf.clear()
        self._speech = self._silence = 0
        self.active = False
        self.det.reset()


def pcm_seconds(pcm: bytes) -> float:
    return len(pcm) / (SAMPLE_RATE * 2)

