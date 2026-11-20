from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from colab_inference import FAILURE_COOLDOWN_SECONDS, RemoteWorker, build_remote_worker, encode_frame
from detector import YoloObstacleDetector, canonical_class_name
from main import app, detector as app_detector


NAMES = {"0": "Nguoi", "1": "Cot"}
HEALTH = {"status": "ok", "ready": True, "model": "m.pt", "runtime": "ultralytics/cuda", "names": NAMES}


class Clock:
    now = 100.0

    def __call__(self) -> float:
        return self.now


def accepts(name: str) -> bool:
    return canonical_class_name(name) is not None


def make_worker(infer, health=None, clock=None) -> RemoteWorker:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return health(request) if health else httpx.Response(200, json=HEALTH)
        return infer(request)

    client = httpx.Client(base_url="http://127.0.0.1:9000", transport=httpx.MockTransport(handler))
    return RemoteWorker("http://127.0.0.1:9000", accepts, client=client, clock=clock or Clock())


def reply(width: int, height: int, xyxy=None, cls=None, conf=None) -> httpx.Response:
    return httpx.Response(200, json={"w": width, "h": height, "xyxy": xyxy or [], "cls": cls or [], "conf": conf or []})


class RemoteWorkerTest(unittest.TestCase):
    def test_not_ready_until_health_passes_and_never_calls_infer(self):
        calls = []
        worker = make_worker(lambda request: calls.append(request) or reply(64, 64))
        self.assertIsNone(worker.infer(Image.new("RGB", (64, 64))))
        self.assertEqual(calls, [])
        self.assertTrue(worker.check_health())
        self.assertTrue(worker.ready)

    def test_boxes_are_scaled_back_to_the_uploaded_image(self):
        worker = make_worker(lambda _: reply(640, 320, [[64.0, 32.0, 128.0, 64.0]], [0], [0.9]))
        worker.check_health()
        result = worker.infer(Image.new("RGB", (1280, 640)))
        self.assertEqual(result.boxes[0].xyxy, (128.0, 64.0, 256.0, 128.0))
        self.assertEqual(result.boxes[0].class_name, "Nguoi")
        self.assertEqual(result.model, "m.pt")

    def test_busy_worker_uses_local_for_one_frame_but_stays_ready(self):
        worker = make_worker(lambda _: httpx.Response(503))
        worker.check_health()
        self.assertIsNone(worker.infer(Image.new("RGB", (64, 64))))
        self.assertTrue(worker.ready)

    def test_timeout_marks_unavailable_until_cooldown_and_health_recovers(self):
        clock = Clock()

        def timeout(request):
            raise httpx.ReadTimeout("slow", request=request)

        worker = make_worker(timeout, clock=clock)
        worker.check_health()
        self.assertIsNone(worker.infer(Image.new("RGB", (64, 64))))
        self.assertFalse(worker.ready)
        self.assertFalse(worker.check_health())
        clock.now += FAILURE_COOLDOWN_SECONDS
        self.assertTrue(worker.check_health())
        self.assertTrue(worker.ready)

    def test_malformed_replies_fall_back_instead_of_raising(self):
        bad = [
            httpx.Response(200, text="not json"),
            httpx.Response(500),
            httpx.Response(200, json=[]),
            reply(10, 10),
            reply(64, 64, [[1, 2, 3]], [0], [0.5]),
            reply(64, 64, [[1, 2, 3, 4]], [9], [0.5]),
            reply(64, 64, [[1, 2, 3, 4]], [0], [1.5]),
            reply(64, 64, [[9, 9, 1, 1]], [0], [0.5]),
            reply(64, 64, [[1, 2, 3, 4]], [0, 1], [0.5]),
        ]
        for response in bad:
            with self.subTest(body=response.content[:40]):
                worker = make_worker(lambda _: response)
                worker.check_health()
                self.assertIsNone(worker.infer(Image.new("RGB", (64, 64))))
                self.assertFalse(worker.ready)

    def test_health_rejects_unready_or_incompatible_workers(self):
        for payload in ({**HEALTH, "ready": False}, {**HEALTH, "names": {"0": "airplane"}}, {"status": "ok"}):
            with self.subTest(payload=payload):
                worker = make_worker(lambda _: reply(64, 64), health=lambda _: httpx.Response(200, json=payload))
                self.assertFalse(worker.check_health())

    def test_connection_failure_is_not_raised(self):
        def refused(request):
            raise httpx.ConnectError("down", request=request)

        worker = make_worker(refused, health=refused)
        self.assertFalse(worker.check_health())
        self.assertIsNone(worker.infer(Image.new("RGB", (64, 64))))

    def test_only_loopback_urls_are_accepted(self):
        self.assertIsNone(build_remote_worker("", accepts))
        self.assertIsNone(build_remote_worker("http://203.0.113.5:9000", accepts))
        self.assertIsNone(build_remote_worker("https://127.0.0.1:9000", accepts))
        worker = build_remote_worker("http://127.0.0.1:9000", accepts)
        self.assertIsNotNone(worker)
        worker.stop()

    def test_large_frames_are_shrunk_to_640(self):
        _, size = encode_frame(Image.new("RGB", (1920, 1080)))
        self.assertEqual(size, (640, 360))
        _, size = encode_frame(Image.new("RGB", (320, 240)))
        self.assertEqual(size, (320, 240))


class DetectorBranchTest(unittest.TestCase):
    def make_detector(self, worker) -> YoloObstacleDetector:
        with patch("detector.detector_setting", side_effect=lambda _key, default: default):
            detector = YoloObstacleDetector()
        detector._remote = worker
        return detector

    def test_colab_objects_have_the_same_shape_as_local_objects(self):
        worker = make_worker(lambda _: reply(64, 64, [[8.0, 8.0, 32.0, 48.0]], [0], [0.9]))
        worker.check_health()
        detector = self.make_detector(worker)
        objects = detector.detect(Image.new("RGB", (64, 64)))
        expected = YoloObstacleDetector._build_object("Nguoi", 0.9, (8.0, 8.0, 32.0, 48.0), 64, 64)
        self.assertEqual(objects, [expected])
        self.assertEqual(objects[0]["class_name"], "person")
        self.assertEqual(detector.last_source()["source"], "colab")

    def test_low_confidence_colab_boxes_are_dropped(self):
        worker = make_worker(lambda _: reply(64, 64, [[8.0, 8.0, 32.0, 48.0]], [0], [0.2]))
        worker.check_health()
        self.assertEqual(self.make_detector(worker).detect(Image.new("RGB", (64, 64))), [])

    def test_failed_colab_call_runs_the_local_model(self):
        worker = make_worker(lambda _: httpx.Response(500))
        worker.check_health()
        detector = self.make_detector(worker)
        with patch.object(detector, "_get_model") as get_model:
            get_model.return_value.predict.return_value = []
            self.assertEqual(detector.detect(Image.new("RGB", (64, 64))), [])
        get_model.return_value.predict.assert_called_once()
        self.assertEqual(detector.last_source()["source"], "local")


class DetectEndpointTest(unittest.TestCase):
    def test_diagnostics_report_the_branch_without_changing_the_contract(self):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buffer, "JPEG")
        source = {"source": "colab", "model": "m.pt", "runtime": "ultralytics/cuda"}
        with patch.object(app_detector, "detect", return_value=[]), patch.object(
            app_detector, "last_source", return_value=source
        ):
            response = TestClient(app).post(
                "/detect?diagnostics=true", files={"image": ("t.jpg", buffer.getvalue(), "image/jpeg")}
            )
        body = json.loads(response.content)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(body), {"success", "image_width", "image_height", "objects", "diagnostics"})
        self.assertEqual(body["diagnostics"]["source"], "colab")
        self.assertEqual(body["diagnostics"]["model"], "m.pt")


if __name__ == "__main__":
    unittest.main()
