"""Optional remote YOLO worker reached through a loopback SSH tunnel.

The worker is only ever a faster path. Every failure here returns ``None`` so the
caller runs the local model, and /detect keeps answering when Colab disappears.
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
from PIL import Image


logger = logging.getLogger("smart-navigation")

MAX_SIDE = 640
JPEG_QUALITY = 85
HEALTH_INTERVAL_SECONDS = 4.0
FAILURE_COOLDOWN_SECONDS = 10.0
REQUEST_TIMEOUT = httpx.Timeout(2.0, connect=0.3, pool=0.1)
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class RemoteBox:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RemoteResult:
    boxes: list[RemoteBox]
    model: str
    runtime: str


def encode_frame(image: Image.Image) -> tuple[bytes, tuple[int, int]]:
    """Shrink to the model's input size so the tunnel carries as little as possible."""
    width, height = image.size
    scale = min(1.0, MAX_SIDE / max(width, height))
    frame = image
    if scale < 1.0:
        frame = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    buffer = io.BytesIO()
    frame.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue(), frame.size


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_boxes(
    payload: Any,
    sent_size: tuple[int, int],
    original_size: tuple[int, int],
    names: dict[int, str],
) -> list[RemoteBox]:
    """Validate the worker's reply and map boxes back onto the uploaded image."""
    if not isinstance(payload, dict) or (payload.get("w"), payload.get("h")) != sent_size:
        raise ValueError("unexpected frame size")
    xyxy, classes, confidences = payload.get("xyxy"), payload.get("cls"), payload.get("conf")
    if not all(isinstance(item, list) for item in (xyxy, classes, confidences)):
        raise ValueError("missing detection arrays")
    if not len(xyxy) == len(classes) == len(confidences):
        raise ValueError("detection arrays differ in length")

    original_width, original_height = original_size
    scale_x = original_width / sent_size[0]
    scale_y = original_height / sent_size[1]
    boxes: list[RemoteBox] = []
    for box, class_id, confidence in zip(xyxy, classes, confidences):
        if not isinstance(box, list) or len(box) != 4 or not all(_finite(value) for value in box):
            raise ValueError("invalid box")
        if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id not in names:
            raise ValueError("unknown class")
        if not _finite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid confidence")
        x1 = min(max(box[0] * scale_x, 0.0), original_width)
        y1 = min(max(box[1] * scale_y, 0.0), original_height)
        x2 = min(max(box[2] * scale_x, 0.0), original_width)
        y2 = min(max(box[3] * scale_y, 0.0), original_height)
        if x2 < x1 or y2 < y1:
            raise ValueError("inverted box")
        boxes.append(RemoteBox(names[class_id], float(confidence), (x1, y1, x2, y2)))
    return boxes


class RemoteWorker:
    """Tracks worker readiness in the background so requests never wait on /health."""

    def __init__(
        self,
        base_url: str,
        accepts_class: Callable[[str], bool],
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._accepts_class = accepts_class
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=REQUEST_TIMEOUT,
            trust_env=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._ready = False
        self._down_until = 0.0
        self._names: dict[int, str] = {}
        self._model = "remote"
        self._runtime = "remote"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready and self._clock() >= self._down_until

    def _set_unavailable(self, reason: str, cooldown: float = 0.0) -> None:
        with self._lock:
            was_ready = self._ready
            self._ready = False
            if cooldown:
                self._down_until = max(self._down_until, self._clock() + cooldown)
        if was_ready:
            logger.warning("Colab worker unavailable, using local model: %s", reason)

    def check_health(self) -> bool:
        try:
            response = self._client.get("/health", timeout=2.0)
            payload = response.json()
            if response.status_code != 200 or payload.get("status") != "ok" or payload.get("ready") is not True:
                raise ValueError("worker is not ready")
            names = {int(key): str(value) for key, value in payload["names"].items()}
            if not names or not all(self._accepts_class(name) for name in names.values()):
                raise ValueError("worker classes do not match this server")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as error:
            self._set_unavailable(type(error).__name__)
            return False

        with self._lock:
            if self._clock() < self._down_until:
                return False
            became_ready = not self._ready
            self._ready = True
            self._names = names
            self._model = str(payload.get("model") or "remote")
            self._runtime = str(payload.get("runtime") or "remote")
        if became_ready:
            logger.info("Colab worker ready: model=%s runtime=%s", self._model, self._runtime)
        return True

    def infer(self, image: Image.Image) -> RemoteResult | None:
        if not self.ready:
            return None
        with self._lock:
            names, model, runtime = self._names, self._model, self._runtime
        try:
            body, sent_size = encode_frame(image)
            response = self._client.post("/infer", content=body, headers={"Content-Type": "image/jpeg"})
            if response.status_code == 503:
                return None  # Busy with another frame: answer this one locally, stay ready.
            response.raise_for_status()
            boxes = parse_boxes(response.json(), sent_size, image.size, names)
        except httpx.PoolTimeout:
            return None
        except (httpx.HTTPError, ValueError, TypeError, OSError) as error:
            self._set_unavailable(type(error).__name__, FAILURE_COOLDOWN_SECONDS)
            return None
        return RemoteResult(boxes, model, runtime)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.check_health()
            self._stop.wait(HEALTH_INTERVAL_SECONDS)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._poll, name="colab-health", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self._client.close()


def build_remote_worker(url: str, accepts_class: Callable[[str], bool]) -> RemoteWorker | None:
    """The worker has no authentication, so only a loopback tunnel endpoint is allowed."""
    url = url.strip()
    if not url:
        return None
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS:
        logger.error("Ignoring COLAB_WORKER_URL: it must be an http loopback address")
        return None
    # httpx logs every request at INFO, which would flood server.log at frame rate.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return RemoteWorker(url, accepts_class)
