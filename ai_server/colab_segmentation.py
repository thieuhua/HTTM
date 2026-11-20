"""Optional walkable segmentation over a loopback tunnel, independent of YOLO."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import logging
import math
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
from PIL import Image

from colab_inference import encode_frame, LOOPBACK_HOSTS


logger = logging.getLogger("smart-navigation")
EXPECTED_CLASSES = ["other", "floor", "sidewalk", "road", "crosswalk", "stairs", "curb", "terrain", "obstacle"]
HAZARDS = {"stairs_ahead", "curb_ahead", "road_ahead", "crosswalk_ahead", "obstacle_ahead"}
HEALTH_INTERVAL_SECONDS = 4.0
FAILURE_COOLDOWN_SECONDS = 10.0
UNAVAILABLE_COOLDOWN_SECONDS = 5.0
CACHE_TTL_SECONDS = 1.5
REQUEST_TIMEOUT = httpx.Timeout(3.0, connect=0.3, pool=0.1)


class SegmentationUnavailable(Exception):
    """Worker is off, incompatible, disconnected, or cooling down."""


class SegmentationBusy(Exception):
    """A frame is in flight or the minimum interval has not elapsed."""


@dataclass(frozen=True)
class SegmentationResult:
    payload: dict[str, Any]
    model: str
    runtime: str
    cached: bool = False
    cache_age_ms: float | None = None


def _finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _fraction(value: Any) -> bool:
    return _finite(value) and 0 <= value <= 1


def parse_reply(payload: Any, sent_size: tuple[int, int], mask: str) -> dict[str, Any]:
    """Validate the worker contract; only explicitly supported fields escape."""
    if (not isinstance(payload, dict)
            or any(type(payload.get(key)) is not int for key in ("w", "h"))
            or (payload["w"], payload["h"]) != sent_size):
        raise ValueError("unexpected frame size")
    if payload.get("classes") != EXPECTED_CLASSES:
        raise ValueError("unexpected classes")
    ratios = payload.get("ratios")
    if (not isinstance(ratios, dict) or set(ratios) != set(EXPECTED_CLASSES)
            or not all(_fraction(value) for value in ratios.values())):
        raise ValueError("invalid class ratios")
    grid = payload.get("grid")
    if not isinstance(grid, list) or len(grid) != 3:
        raise ValueError("invalid grid")
    for row in grid:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("invalid grid row")
        for cell in row:
            if (not isinstance(cell, dict)
                    or not all(_fraction(cell.get(key)) for key in ("walkable", "caution", "blocked"))
                    or cell.get("dominant") not in EXPECTED_CLASSES):
                raise ValueError("invalid grid cell")
    if type(payload.get("path_clear")) is not bool:
        raise ValueError("invalid path_clear")
    if payload.get("suggest") not in ("straight", "left", "right", "stop"):
        raise ValueError("invalid suggestion")
    hazards = payload.get("hazards")
    if not isinstance(hazards, list) or not all(isinstance(item, str) and item in HAZARDS for item in hazards):
        raise ValueError("invalid hazards")
    if not _finite(payload.get("latency_ms")) or payload["latency_ms"] < 0:
        raise ValueError("invalid latency")
    result = {key: payload[key] for key in ("classes", "ratios", "grid", "path_clear", "suggest", "hazards")}
    if mask != "none":
        w, h = payload.get("mask_w"), payload.get("mask_h")
        if type(w) is not int or type(h) is not int or not (0 < w <= sent_size[0] and 0 < h <= sent_size[1]):
            raise ValueError("invalid mask dimensions")
        result.update(mask_w=w, mask_h=h)
        if mask == "png":
            if not isinstance(payload.get("mask_png_b64"), str) or not payload["mask_png_b64"]:
                raise ValueError("missing PNG mask")
            result["mask_png_b64"] = payload["mask_png_b64"]
        else:
            rle = payload.get("mask_rle")
            if not isinstance(rle, dict) or rle.get("size") != [h, w] or rle.get("order") != "C":
                raise ValueError("invalid RLE shape")
            values, counts = rle.get("values"), rle.get("counts")
            if (not isinstance(values, list) or not isinstance(counts, list) or len(values) != len(counts)
                    or not all(type(value) is int and 0 <= value < len(EXPECTED_CLASSES) for value in values)
                    or not all(type(count) is int and count > 0 for count in counts)
                    or sum(counts) != w * h):
                raise ValueError("invalid RLE runs")
            result["mask_rle"] = rle
    return deepcopy(result)


class SegmentationWorker:
    """Share one keep-alive client between health polling and guarded inference."""

    def __init__(self, base_url: str, client: httpx.Client | None = None,
                 clock: Callable[[], float] = time.monotonic, min_interval_ms: float = 500) -> None:
        if not _finite(min_interval_ms) or min_interval_ms < 0:
            raise ValueError("min_interval_ms must be finite and nonnegative")
        self._client = client if client is not None else httpx.Client(
            base_url=base_url.rstrip("/"), timeout=REQUEST_TIMEOUT, trust_env=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2, keepalive_expiry=30.0),
        )
        self._clock = clock
        self._min_interval = min_interval_ms / 1000
        self._lock = threading.Lock()
        self._in_flight = threading.Lock()
        self._ready = False
        self._down_until = 0.0
        self._generation = 0
        self._model = "remote"
        self._runtime = "remote"
        self._last_started = -math.inf
        self._cache: dict[str, tuple[float, SegmentationResult]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready and not self._stop.is_set() and self._clock() >= self._down_until

    def _set_unavailable(self, reason: str, cooldown: float = 0.0) -> None:
        with self._lock:
            was_ready = self._ready
            self._ready = False
            self._generation += 1
            self._cache.clear()
            self._down_until = max(self._down_until, self._clock() + cooldown)
        if was_ready:
            logger.warning("Segmentation worker unavailable: %s", reason)

    def check_health(self) -> bool:
        with self._lock:
            if self._stop.is_set() or self._clock() < self._down_until:
                return False
            generation = self._generation
        try:
            response = self._client.get("/health")
            payload = response.json()
            if (response.status_code != 200 or not isinstance(payload, dict)
                    or payload.get("status") != "ok" or payload.get("ready") is not True
                    or payload.get("task") != "segmentation" or payload.get("classes") != EXPECTED_CLASSES):
                raise ValueError("incompatible or unready segmentation worker")
            runtime = payload.get("runtime")
            if (not isinstance(runtime, dict) or not isinstance(runtime.get("device"), str)
                    or not runtime["device"] or type(runtime.get("fp16")) is not bool):
                raise ValueError("invalid segmentation runtime")
            model = payload.get("model")
            if not isinstance(model, str) or not model:
                raise ValueError("missing segmentation model")
            runtime_name = runtime["device"] + ("/fp16" if runtime["fp16"] else "/fp32")
        except httpx.PoolTimeout:
            return False  # Pool contention does not invalidate readiness or cache.
        except (httpx.HTTPError, ValueError, TypeError) as error:
            self._set_unavailable(type(error).__name__)
            return False
        with self._lock:
            if self._stop.is_set() or generation != self._generation or self._clock() < self._down_until:
                return False
            became_ready = not self._ready
            self._ready = True
            self._model, self._runtime = model, runtime_name
        if became_ready:
            logger.info("Segmentation worker ready: model=%s runtime=%s", model, runtime_name)
        return True

    def segment(self, image: Image.Image, mask: str = "none") -> SegmentationResult:
        if mask not in ("none", "png", "rle"):
            raise ValueError("invalid mask mode")
        if not self.ready:
            raise SegmentationUnavailable()
        # Never queue frames on threadpool threads also needed by /detect.
        if not self._in_flight.acquire(blocking=False):
            raise SegmentationBusy()
        try:
            with self._lock:
                now = self._clock()
                if not self._ready or self._stop.is_set() or now < self._down_until:
                    raise SegmentationUnavailable()
                if now - self._last_started < self._min_interval:
                    entry = self._cache.get(mask)
                    if entry is not None and now - entry[0] < CACHE_TTL_SECONDS:
                        return replace(deepcopy(entry[1]), cached=True, cache_age_ms=(now - entry[0]) * 1000)
                    raise SegmentationBusy()
                self._last_started = now
                generation, model, runtime = self._generation, self._model, self._runtime
            try:
                body, sent_size = encode_frame(image.convert("RGB"))
                response = self._client.post("/segment", params={"mask": mask}, content=body,
                                             headers={"Content-Type": "image/jpeg"})
                if response.status_code == 503:
                    self._set_unavailable("worker 503", UNAVAILABLE_COOLDOWN_SECONDS)
                    raise SegmentationUnavailable()
                response.raise_for_status()
                payload = parse_reply(response.json(), sent_size, mask)
            except httpx.PoolTimeout as error:
                raise SegmentationBusy() from error
            except (httpx.HTTPError, ValueError, TypeError, OSError) as error:
                self._set_unavailable(type(error).__name__, FAILURE_COOLDOWN_SECONDS)
                raise SegmentationUnavailable() from error
            result = SegmentationResult(payload, model, runtime)
            with self._lock:
                # A failed health poll invalidates even a concurrently successful frame.
                if not self._ready or generation != self._generation or self._stop.is_set():
                    raise SegmentationUnavailable()
                self._cache[mask] = (self._clock(), deepcopy(result))
            return result
        finally:
            self._in_flight.release()

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.check_health()
            self._stop.wait(HEALTH_INTERVAL_SECONDS)

    def start(self) -> None:
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._poll, name="segmentation-health", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._set_unavailable("stopped")
        if self._thread is not None:
            self._thread.join(timeout=4)
            self._thread = None
        self._client.close()


def build_segmentation_worker(url: str, *, min_interval_ms: float = 500) -> SegmentationWorker | None:
    """Only an unauthenticated HTTP loopback tunnel may receive camera frames."""
    url = url.strip()
    if not url:
        return None
    try:
        parts = urlsplit(url)
        if (parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS
                or parts.username is not None or parts.password is not None):
            raise ValueError("not an http loopback address")
        parts.port  # Validate malformed ports before httpx constructs the client.
    except ValueError:
        logger.error("Ignoring COLAB_SEG_URL: it must be an http loopback address")
        return None
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return SegmentationWorker(url, min_interval_ms=min_interval_ms)
