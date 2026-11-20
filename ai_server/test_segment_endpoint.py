"""Public /segment contract and /detect independence, using no network sockets."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import os
import threading
import unittest
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

import main
from colab_segmentation import SegmentationBusy, SegmentationUnavailable, build_segmentation_worker
from test_colab_segmentation import Clock, FIXTURE, make_worker


class SegmentEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.clock = Clock()
        self.worker = make_worker(clock=self.clock)
        self.worker.check_health()
        self.addCleanup(self.worker.stop)
        self.addCleanup(self.client.close)
        worker_patch = patch.object(main, "segmentation_worker", self.worker)
        worker_patch.start()
        self.addCleanup(worker_patch.stop)
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48)).save(buffer, "PNG")
        self.png = buffer.getvalue()

    def upload(self, data=None, query="", client=None):
        return (client or self.client).post("/segment" + query, files={
            "image": ("frame.png", self.png if data is None else data, "image/png")})

    def test_success_has_exact_shared_contract_and_no_diagnostics_by_default(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"success", "image_width", "image_height", "classes", "ratios",
                                     "grid", "path_clear", "suggest", "hazards", "cached"})
        self.assertIs(body["success"], True)
        self.assertIs(body["cached"], False)
        self.assertEqual((body["image_width"], body["image_height"]), (64, 48))
        for key in ("classes", "ratios", "grid", "path_clear", "suggest", "hazards"):
            self.assertEqual(body[key], FIXTURE["none"][key])

    def test_diagnostics_use_formatted_runtime_and_measured_latency(self):
        response = self.upload(query="?diagnostics=true")
        self.assertEqual(response.status_code, 200)
        diagnostics = response.json()["diagnostics"]
        self.assertEqual(set(diagnostics), {"source", "model", "runtime", "inference_ms"})
        self.assertEqual(diagnostics["source"], "colab")
        self.assertEqual(diagnostics["model"], "best")
        self.assertEqual(diagnostics["runtime"], "cuda/fp16")
        self.assertIsInstance(diagnostics["inference_ms"], float)
        self.assertGreaterEqual(diagnostics["inference_ms"], 0)

    def test_masks_are_requested_and_forwarded_verbatim(self):
        seen = []

        def infer(request):
            mode = request.url.params["mask"]
            seen.append(mode)
            return httpx.Response(200, json=FIXTURE[mode])

        worker = make_worker(infer=infer, min_interval_ms=0)
        self.addCleanup(worker.stop)
        worker.check_health()
        with patch.object(main, "segmentation_worker", worker):
            for mode, field in (("png", "mask_png_b64"), ("rle", "mask_rle")):
                response = self.upload(query="?mask=" + mode)
                self.assertEqual(response.status_code, 200)
                body = response.json()
                for key in ("mask_w", "mask_h", field):
                    self.assertEqual(body[key], FIXTURE[mode][key])
                self.assertNotIn("mask_rle" if mode == "png" else "mask_png_b64", body)
        self.assertEqual(seen, ["png", "rle"])

    def test_cached_response_uses_this_upload_dimensions_and_optional_cache_age(self):
        self.assertEqual(self.upload().status_code, 200)
        self.clock.now = 100.25
        buffer = io.BytesIO()
        Image.new("RGB", (19, 11)).save(buffer, "PNG")
        body = self.upload(buffer.getvalue(), "?diagnostics=true").json()
        self.assertIs(body["cached"], True)
        self.assertEqual((body["image_width"], body["image_height"]), (19, 11))
        self.assertEqual(body["diagnostics"]["cache_age_ms"], 250)
        body = self.upload(buffer.getvalue()).json()
        self.assertIs(body["cached"], True)
        self.assertNotIn("diagnostics", body)

    def test_exif_orientation_and_large_upload_dimensions(self):
        for size, orientation, expected in (((64, 48), 6, (48, 64)), ((1280, 640), None, (1280, 640))):
            with self.subTest(size=size):
                buffer = io.BytesIO()
                image = Image.new("RGB", size)
                exif = image.getexif()
                if orientation:
                    exif[274] = orientation
                image.save(buffer, "JPEG", exif=exif)

                def infer(request):
                    with Image.open(io.BytesIO(request.content)) as sent:
                        return httpx.Response(200, json={**FIXTURE["none"], "w": sent.width, "h": sent.height})

                worker = make_worker(infer=infer)
                self.addCleanup(worker.stop)
                worker.check_health()
                with patch.object(main, "segmentation_worker", worker):
                    response = self.upload(buffer.getvalue())
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual((body["image_width"], body["image_height"]), expected)

    def test_missing_image_and_invalid_mask_return_422(self):
        self.assertEqual(self.client.post("/segment").status_code, 422)
        with patch.object(self.worker, "segment") as infer:
            self.assertEqual(self.upload(query="?mask=jpeg").status_code, 422)
            self.assertEqual(self.upload(query="?diagnostics=nonsense").status_code, 422)
            infer.assert_not_called()

    def test_empty_corrupt_and_truncated_images_match_detect_errors_and_logs(self):
        with patch.object(self.worker, "segment") as infer:
            for data in (b"", b"not an image", self.png[:20]):
                with self.subTest(data=data):
                    with self.assertLogs("smart-navigation", level="WARNING") as seg_logs:
                        response = self.upload(data)
                    with self.assertLogs("smart-navigation", level="WARNING") as detect_logs:
                        detect = self.client.post("/detect", files={"image": ("frame.png", data, "image/png")})
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json(), detect.json())
                    self.assertEqual(seg_logs.output, detect_logs.output)
            infer.assert_not_called()

    def test_byte_limit_matches_detect_including_log_line(self):
        with patch.dict(os.environ, {"DETECT_MAX_BYTES": "50"}), patch.object(self.worker, "segment") as infer:
            with self.assertLogs("smart-navigation", level="WARNING") as seg_logs:
                response = self.upload()
            with self.assertLogs("smart-navigation", level="WARNING") as detect_logs:
                detect = self.client.post("/detect", files={"image": ("frame.png", self.png, "image/png")})
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json(), {"detail": "Uploaded image exceeds maximum size."})
            self.assertEqual(response.json(), detect.json())
            self.assertEqual(seg_logs.output, detect_logs.output)
            infer.assert_not_called()

    def test_pixel_limit_returns_413_before_loading(self):
        buffer = io.BytesIO()
        Image.new("L", (4000, 3001)).save(buffer, "PNG")
        with patch.object(Image.Image, "load") as load, patch.object(self.worker, "segment") as infer:
            response = self.upload(buffer.getvalue())
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json(), {"detail": "Image exceeds maximum pixel limit."})
            load.assert_not_called()
            infer.assert_not_called()

    def test_decompression_bomb_error_matches_detect(self):
        with patch.object(main.Image, "open", side_effect=Image.DecompressionBombError("too large")):
            response = self.upload()
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"detail": "Image exceeds maximum pixel limit."})

    def test_empty_url_disables_segmentation_and_detect_still_returns_200(self):
        with patch.object(main, "segmentation_worker", build_segmentation_worker("")), patch.object(
                main.detector, "detect", return_value=[]):
            response = self.upload()
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "Segmentation worker unavailable."})
            detect = self.client.post("/detect", files={"image": ("frame.png", self.png, "image/png")})
            self.assertEqual(detect.status_code, 200)
            self.assertEqual(detect.json()["objects"], [])

    def test_unready_worker_and_worker_503_return_unavailable(self):
        unready = make_worker()
        down = make_worker(infer=lambda _: httpx.Response(503))
        self.addCleanup(unready.stop)
        self.addCleanup(down.stop)
        down.check_health()
        for worker in (unready, down):
            with self.subTest(worker=worker), patch.object(main, "segmentation_worker", worker):
                response = self.upload()
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json(), {"detail": "Segmentation worker unavailable."})
                self.assertFalse(worker.ready)
                self.assertEqual(self.upload().json(), response.json())

    def test_busy_and_unexpected_errors_have_stable_details(self):
        for error, code, detail in ((SegmentationBusy(), 503, "Segmentation worker busy."),
                                    (SegmentationUnavailable(), 503, "Segmentation worker unavailable."),
                                    (RuntimeError("private internal details"), 500, "Segmentation failed.")):
            with self.subTest(error=error), patch.object(self.worker, "segment", side_effect=error):
                response = self.upload()
                self.assertEqual(response.status_code, code)
                self.assertEqual(response.json(), {"detail": detail})

    def test_min_interval_without_matching_mask_cache_is_busy(self):
        self.assertEqual(self.upload().status_code, 200)
        response = self.upload(query="?mask=png")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Segmentation worker busy."})

    def test_assistant_boundary_does_not_authenticate_segment(self):
        with patch("assistant_router.guard.require_enabled", side_effect=AssertionError("assistant checked")), patch(
                "assistant_router.guard.authenticate", side_effect=AssertionError("assistant auth checked")):
            self.assertEqual(self.upload().status_code, 200)

    def test_health_contract_is_unchanged_when_segmentation_is_off(self):
        with patch.object(main, "segmentation_worker", None):
            self.assertEqual(self.client.get("/health").json(), {
                "status": "ok", "routing_profile": "pedestrian", "api_version": "2"})

    def test_detect_still_returns_200_while_a_segment_request_is_in_flight(self):
        entered, release = threading.Event(), threading.Event()

        def infer(_):
            entered.set()
            if not release.wait(4):
                raise RuntimeError("test release timeout")
            return httpx.Response(200, json=FIXTURE["none"])

        worker = make_worker(infer=infer)
        self.addCleanup(worker.stop)
        worker.check_health()
        with patch.object(main, "segmentation_worker", worker), patch.object(main.detector, "detect", return_value=[]):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.upload)
                try:
                    self.assertTrue(entered.wait(2))
                    detect = pool.submit(self.client.post, "/detect", files={"image": ("frame.png", self.png, "image/png")})
                    self.assertEqual(detect.result(timeout=2).status_code, 200)
                    busy = pool.submit(self.upload).result(timeout=1)
                    self.assertEqual(busy.status_code, 503)
                    self.assertEqual(busy.json(), {"detail": "Segmentation worker busy."})
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=2).status_code, 200)

    def test_segmentation_inference_runs_on_a_threadpool_thread(self):
        async def exercise():
            event_thread = threading.get_ident()
            original = self.worker.segment

            def segment(*args):
                self.assertNotEqual(threading.get_ident(), event_thread)
                return original(*args)

            with patch.object(self.worker, "segment", side_effect=segment):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                    response = await client.post("/segment", files={"image": ("f.png", self.png, "image/png")})
                    self.assertEqual(response.status_code, 200)

        asyncio.run(exercise())

    def test_lifespan_starts_and_stops_segmentation_with_detector(self):
        worker = Mock()
        with patch.object(main, "segmentation_worker", worker), patch.dict(os.environ, {"YOLO_PRELOAD": "0"}), patch.object(
                main.detector, "start_remote") as start, patch.object(main.detector, "stop_remote") as stop:
            with TestClient(main.app):
                worker.start.assert_called_once_with()
                start.assert_called_once_with()
                worker.stop.assert_not_called()
            worker.stop.assert_called_once_with()
            stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
