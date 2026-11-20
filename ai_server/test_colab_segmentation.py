"""Offline worker contract, readiness, and concurrency tests (no sockets)."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from colab_segmentation import (
    CACHE_TTL_SECONDS, EXPECTED_CLASSES, FAILURE_COOLDOWN_SECONDS,
    HEALTH_INTERVAL_SECONDS, REQUEST_TIMEOUT, SegmentationBusy, SegmentationUnavailable,
    SegmentationWorker, UNAVAILABLE_COOLDOWN_SECONDS, build_segmentation_worker, parse_reply,
)


# Captured from the real worker_seg.create_app with an injected numpy FakeModel,
# using the read-only walkable-seg checkout and its virtualenv. No real weights.
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "walkable_worker_reply.json").read_text(encoding="utf-8"))
HEALTH = {**FIXTURE["health"], "runtime": {"device": "cuda", "fp16": True}}


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make_worker(infer=None, health=None, clock=None, min_interval_ms=500):
    def handler(request):
        if request.url.path == "/health":
            return health(request) if health else httpx.Response(200, json=HEALTH)
        if infer:
            return infer(request)
        return httpx.Response(200, json=FIXTURE[request.url.params["mask"]])

    client = httpx.Client(base_url="http://127.0.0.1:9001", transport=httpx.MockTransport(handler))
    return SegmentationWorker("http://127.0.0.1:9001", client=client,
                              clock=clock or Clock(), min_interval_ms=min_interval_ms)


class SegmentationWorkerTests(unittest.TestCase):
    def worker(self, **kwargs):
        worker = make_worker(**kwargs)
        self.addCleanup(worker.stop)
        return worker

    def frame(self):
        return Image.new("RGB", (64, 48))

    def test_unready_never_sends_a_frame(self):
        calls = []
        worker = self.worker(infer=lambda request: calls.append(request))
        self.assertFalse(worker.ready)
        with self.assertRaises(SegmentationUnavailable):
            worker.segment(self.frame())
        self.assertEqual(calls, [])

    def test_health_accepts_exact_classes_and_formats_runtime_dict(self):
        for fp16, device, expected in ((True, "cuda", "cuda/fp16"), (False, "cpu", "cpu/fp32")):
            with self.subTest(runtime=expected):
                worker = self.worker(health=lambda _: httpx.Response(200, json={
                    **HEALTH, "runtime": {"device": device, "fp16": fp16}}))
                self.assertTrue(worker.check_health())
                self.assertTrue(worker.ready)
                result = worker.segment(self.frame())
                self.assertEqual(result.model, "best")
                self.assertEqual(result.runtime, expected)

    def test_health_rejects_unready_missing_reordered_and_wrong_classes(self):
        missing = deepcopy(HEALTH)
        del missing["classes"]
        invalid = [
            {**HEALTH, "ready": False}, {**HEALTH, "ready": 1}, missing,
            {**HEALTH, "classes": EXPECTED_CLASSES[::-1]},
            {**HEALTH, "classes": EXPECTED_CLASSES[:-1]},
            {**HEALTH, "status": "down"}, {**HEALTH, "task": "detection"},
            {**HEALTH, "runtime": "cuda/fp16"},
            {**HEALTH, "runtime": {"device": "cuda", "fp16": "true"}},
            {**HEALTH, "model": None}, [], None,
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                worker = self.worker(health=lambda _: httpx.Response(200, json=payload))
                self.assertFalse(worker.check_health())
                self.assertFalse(worker.ready)
        for status in (201, 503):
            worker = self.worker(health=lambda _: httpx.Response(status, json=HEALTH))
            self.assertFalse(worker.check_health())

    def test_real_worker_fixture_all_mask_modes_are_forwarded_verbatim(self):
        for mode in ("none", "png", "rle"):
            with self.subTest(mask=mode):
                worker = self.worker()
                worker.check_health()
                result = worker.segment(self.frame(), mode)
                self.assertFalse(result.cached)
                expected = {key: value for key, value in FIXTURE[mode].items()
                            if key not in ("w", "h", "latency_ms")}
                self.assertEqual(result.payload, expected)

    def test_encode_frame_is_reused_and_raw_jpeg_is_resized(self):
        def infer(request):
            self.assertEqual(request.headers["content-type"], "image/jpeg")
            self.assertEqual(request.url.path, "/segment")
            self.assertEqual(request.url.params["mask"], "none")
            with Image.open(io.BytesIO(request.content)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (640, 320))
            return httpx.Response(200, json={**FIXTURE["none"], "w": 640, "h": 320})
        worker = self.worker(infer=infer)
        worker.check_health()
        worker.segment(Image.new("RGBA", (1280, 640)))

    def test_malformed_replies_are_rejected_and_cool_down_for_ten_seconds(self):
        mutations = [
            lambda p: p.update(w=63), lambda p: p.update(w=64.0),
            lambda p: p.update(classes=EXPECTED_CLASSES[::-1]),
            lambda p: p.update(grid=[]), lambda p: p["grid"].__setitem__(0, []),
            lambda p: p["grid"][0].__setitem__(0, None),
            lambda p: p["grid"][0][0].pop("walkable"),
            lambda p: p["grid"][0][0].update(walkable=float("nan")),
            lambda p: p["grid"][0][0].update(caution=-0.1),
            lambda p: p["grid"][0][0].update(blocked=1.1),
            lambda p: p["grid"][0][0].update(dominant="sky"),
            lambda p: p["ratios"].pop("floor"),
            lambda p: p["ratios"].update(floor=float("inf")),
            lambda p: p["ratios"].update(floor=float("nan")),
            lambda p: p["ratios"].update(floor=10 ** 400),
            lambda p: p["ratios"].update(floor=-1),
            lambda p: p["ratios"].update(floor=2),
            lambda p: p["ratios"].update(floor=True),
            lambda p: p.update(suggest="run"), lambda p: p.update(suggest=[]),
            lambda p: p.update(hazards=["sky_ahead"]), lambda p: p.update(hazards=[[]]),
            lambda p: p.update(hazards="stairs_ahead"), lambda p: p.update(path_clear=1),
            lambda p: p.update(latency_ms=-1), lambda p: p.pop("latency_ms"),
        ]
        for mutate in mutations:
            payload = deepcopy(FIXTURE["none"])
            mutate(payload)
            # httpx's JSON encoder rejects non-finite numbers; test malformed wire JSON directly.
            response = httpx.Response(200, text=json.dumps(payload))
            with self.subTest(payload=payload):
                clock = Clock()
                worker = self.worker(infer=lambda _: response, clock=clock)
                worker.check_health()
                with self.assertRaises(SegmentationUnavailable) as error:
                    worker.segment(self.frame())
                self.assertIsInstance(error.exception.__cause__, ValueError)
                self.assertFalse(worker.ready)
                clock.now += FAILURE_COOLDOWN_SECONDS - 1
                self.assertFalse(worker.check_health())
                clock.now += 1
                self.assertTrue(worker.check_health())

    def test_json_errors_and_network_errors_mark_unavailable_until_recovery(self):
        def timeout(request):
            raise httpx.ReadTimeout("slow", request=request)

        def refused(request):
            raise httpx.ConnectError("down", request=request)

        cases = [lambda _: httpx.Response(200, text="not json"),
                 lambda _: httpx.Response(200, json=[]),
                 lambda _: httpx.Response(500), timeout, refused]
        for infer in cases:
            with self.subTest(infer=infer):
                clock = Clock()
                worker = self.worker(infer=infer, clock=clock)
                worker.check_health()
                with self.assertRaises(SegmentationUnavailable):
                    worker.segment(self.frame())
                self.assertFalse(worker.ready)
                clock.now += FAILURE_COOLDOWN_SECONDS
                self.assertTrue(worker.check_health())

    def test_health_network_failure_and_non_json_are_unavailable(self):
        def refused(request):
            raise httpx.ConnectError("down", request=request)
        for health in (refused, lambda _: httpx.Response(200, text="not json")):
            worker = self.worker(health=health)
            self.assertFalse(worker.check_health())
            with self.assertRaises(SegmentationUnavailable):
                worker.segment(self.frame())

    def test_worker_503_has_five_second_cooldown_and_clears_cache(self):
        clock = Clock()
        responses = [httpx.Response(200, json=FIXTURE["none"]), httpx.Response(503),
                     httpx.Response(200, json=FIXTURE["none"])]
        worker = self.worker(infer=lambda _: responses.pop(0), clock=clock, min_interval_ms=1000)
        worker.check_health()
        worker.segment(self.frame())
        clock.now += 1
        with self.assertRaises(SegmentationUnavailable):
            worker.segment(self.frame())
        self.assertFalse(worker.ready)
        self.assertEqual(worker._cache, {})
        clock.now += UNAVAILABLE_COOLDOWN_SECONDS - 1
        self.assertFalse(worker.check_health())
        clock.now += 1
        self.assertTrue(worker.check_health())
        self.assertFalse(worker.segment(self.frame()).cached)

    def test_pool_timeout_is_busy_and_preserves_readiness_and_cache(self):
        def pool_timeout(request):
            raise httpx.PoolTimeout("pool busy", request=request)
        worker = self.worker(infer=pool_timeout)
        worker.check_health()
        with self.assertRaises(SegmentationBusy):
            worker.segment(self.frame())
        self.assertTrue(worker.ready)
        # The health thread also shares the pool: contention must not erase state.
        worker = self.worker()
        worker.check_health()
        worker.segment(self.frame())
        with patch.object(worker._client, "get", side_effect=httpx.PoolTimeout("pool busy")):
            self.assertFalse(worker.check_health())
        self.assertTrue(worker.ready)
        self.assertTrue(worker.segment(self.frame()).cached)

    def test_minimum_interval_cache_age_and_new_inference_at_boundary(self):
        clock = Clock()
        calls = []
        worker = self.worker(infer=lambda request: calls.append(request) or httpx.Response(200, json=FIXTURE["none"]), clock=clock)
        worker.check_health()
        fresh = worker.segment(self.frame())
        clock.now = 100.25
        cached = worker.segment(Image.new("RGB", (17, 9)))
        self.assertTrue(cached.cached)
        self.assertEqual(cached.cache_age_ms, 250)
        self.assertEqual(cached.payload, fresh.payload)
        self.assertEqual(len(calls), 1)
        # Caller mutation must not poison the cached reply.
        cached.payload["grid"][0][0]["dominant"] = "floor"
        self.assertEqual(worker.segment(self.frame()).payload, fresh.payload)
        clock.now = 100.5
        self.assertFalse(worker.segment(self.frame()).cached)
        self.assertEqual(len(calls), 2)

    def test_cache_ttl_is_strictly_less_than_one_point_five_seconds(self):
        clock = Clock()
        worker = self.worker(clock=clock, min_interval_ms=3000)
        worker.check_health()
        worker.segment(self.frame())
        clock.now = 101.49
        self.assertTrue(worker.segment(self.frame()).cached)
        clock.now = 100 + CACHE_TTL_SECONDS
        with self.assertRaises(SegmentationBusy):
            worker.segment(self.frame())
        clock.now = 103
        self.assertFalse(worker.segment(self.frame()).cached)

    def test_cache_is_keyed_by_mask_and_missing_mode_is_busy(self):
        clock = Clock()
        worker = self.worker(clock=clock)
        worker.check_health()
        worker.segment(self.frame(), "none")
        with self.assertRaises(SegmentationBusy):
            worker.segment(self.frame(), "png")
        clock.now = 100.5
        worker.segment(self.frame(), "png")
        with self.assertRaises(SegmentationBusy):
            worker.segment(self.frame(), "rle")
        self.assertTrue(worker.segment(self.frame(), "none").cached)
        self.assertNotIn("mask_w", worker.segment(self.frame(), "none").payload)
        self.assertIn("mask_png_b64", worker.segment(self.frame(), "png").payload)

    def test_health_failure_clears_even_fresh_cached_result(self):
        state = {"ready": True}
        worker = self.worker(health=lambda _: httpx.Response(200, json={**HEALTH, **state}))
        worker.check_health()
        worker.segment(self.frame())
        state["ready"] = False
        self.assertFalse(worker.check_health())
        self.assertEqual(worker._cache, {})
        state["ready"] = True
        worker.check_health()
        with self.assertRaises(SegmentationBusy):
            worker.segment(self.frame())  # Fresh old cache must not leak after recovery.

    def test_one_in_flight_request_never_blocks_the_second(self):
        entered, release = threading.Event(), threading.Event()

        def infer(_):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test release timeout")
            return httpx.Response(200, json=FIXTURE["none"])

        worker = self.worker(infer=infer)
        worker.check_health()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(worker.segment, self.frame())
            try:
                self.assertTrue(entered.wait(2))
                second = pool.submit(worker.segment, self.frame())
                with self.assertRaises(SegmentationBusy):
                    second.result(timeout=1)
            finally:
                release.set()
            self.assertFalse(first.result(timeout=2).cached)

    def test_health_failure_during_inference_does_not_repopulate_cache(self):
        state = {"ready": True}
        worker = None

        def infer(_):
            state["ready"] = False
            self.assertFalse(worker.check_health())
            state["ready"] = True
            self.assertTrue(worker.check_health())
            return httpx.Response(200, json=FIXTURE["none"])

        worker = self.worker(infer=infer, health=lambda _: httpx.Response(200, json={**HEALTH, **state}))
        worker.check_health()
        with self.assertRaises(SegmentationUnavailable):
            worker.segment(self.frame())
        self.assertEqual(worker._cache, {})

    def test_invalid_masks_are_validation_errors(self):
        for mode in ("png", "rle"):
            payload = deepcopy(FIXTURE[mode])
            payload["mask_w"] = 0
            with self.assertRaises(ValueError):
                parse_reply(payload, (64, 48), mode)
        for changes in ({"counts": [-1]}, {"values": [9]}, {"size": [64, 48]}, {"order": "F"}):
            payload = deepcopy(FIXTURE["rle"])
            payload["mask_rle"].update(changes)
            with self.assertRaises(ValueError):
                parse_reply(payload, (64, 48), "rle")
        payload = deepcopy(FIXTURE["png"])
        del payload["mask_png_b64"]
        with self.assertRaises(ValueError):
            parse_reply(payload, (64, 48), "png")

    def test_factory_rejects_empty_public_https_and_malformed_urls(self):
        for url in ("", "  ", "http://203.0.113.5:9001", "https://127.0.0.1:9001",
                    "http://[bad", "http://localhost:bad", "http://localhost@evil.test:9001"):
            with self.subTest(url=url):
                self.assertIsNone(build_segmentation_worker(url))

    def test_one_client_has_required_timeout_and_keepalive_and_is_closed(self):
        for url in ("http://127.0.0.1:9001", "http://localhost:9001", "http://[::1]:9001"):
            client = httpx.Client(base_url=url, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=HEALTH)))
            with patch("colab_segmentation.httpx.Client", return_value=client) as constructor:
                worker = build_segmentation_worker(url)
                self.assertTrue(worker.check_health())
                constructor.assert_called_once()
                kwargs = constructor.call_args.kwargs
                self.assertFalse(kwargs["trust_env"])
                self.assertEqual(kwargs["limits"].keepalive_expiry, 30)
                self.assertEqual(kwargs["timeout"], httpx.Timeout(3.0, connect=0.3, pool=0.1))
                self.assertIs(kwargs["timeout"], REQUEST_TIMEOUT)
                worker.stop()
                self.assertTrue(client.is_closed)

    def test_start_polls_and_stop_closes_the_shared_client(self):
        polled = threading.Event()
        worker = self.worker(health=lambda _: polled.set() or httpx.Response(200, json=HEALTH))
        self.assertEqual(HEALTH_INTERVAL_SECONDS, 4)
        worker.start()
        thread = worker._thread
        worker.start()
        self.assertIs(worker._thread, thread)
        self.assertTrue(polled.wait(2))
        worker.stop()
        self.assertFalse(thread.is_alive())
        self.assertFalse(worker.ready)
        self.assertTrue(worker._client.is_closed)


if __name__ == "__main__":
    unittest.main()
