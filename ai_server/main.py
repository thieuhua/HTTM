"""FastAPI entry point for the Smart Navigation MVP."""

from __future__ import annotations

import io
import logging
import math
import os
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from detector import YoloObstacleDetector, detector_setting
from colab_segmentation import build_segmentation_worker, SegmentationUnavailable, SegmentationBusy
from geocoder import GeocodeError, GeocodeServiceError, geocode, reverse_geocode
from routing import RouteError, plan_route
from assistant_router import router as assistant_router, AssistantBoundary, error_response
from assistant_guard import AssistantError


DEFAULT_DETECT_MAX_BYTES = 5 * 1024 * 1024
DETECT_MAX_PIXELS = 12_000_000
Image.MAX_IMAGE_PIXELS = DETECT_MAX_PIXELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-navigation")

detector = YoloObstacleDetector()
try:
    seg_min_interval_ms = float(detector_setting("SEG_MIN_INTERVAL_MS", "500"))
    if not math.isfinite(seg_min_interval_ms) or seg_min_interval_ms < 0:
        raise ValueError("invalid interval")
except ValueError:
    logger.error("Invalid SEG_MIN_INTERVAL_MS; using 500")
    seg_min_interval_ms = 500.0
segmentation_worker = build_segmentation_worker(
    detector_setting("COLAB_SEG_URL", ""), min_interval_ms=seg_min_interval_ms,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Loading the model takes seconds, longer than the app's 4 s /detect
    # deadline, so do it before the first frame arrives instead of on it.
    if os.getenv("YOLO_PRELOAD", "1") != "0":
        threading.Thread(target=detector.warm_up, name="yolo-preload", daemon=True).start()
    detector.start_remote()
    if segmentation_worker is not None:
        segmentation_worker.start()
    try:
        yield
    finally:
        if segmentation_worker is not None:
            segmentation_worker.stop()
        detector.stop_remote()


def run_detection(image: Image.Image) -> tuple[list[dict[str, object]], dict[str, str]]:
    # last_source() is per thread, so read it on the worker thread that ran detect().
    objects = detector.detect(image)
    return objects, detector.last_source()


app = FastAPI(title="Smart Navigation AI Server", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(assistant_router)
app.add_middleware(AssistantBoundary)


@app.exception_handler(AssistantError)
async def assistant_failure(request, error):
    return error_response(error)


@app.exception_handler(RequestValidationError)
async def validation_failure(request, error):
    if request.url.path.startswith('/assistant/'):
        return JSONResponse({"error": {"code": "invalid_request", "message": "Dữ liệu trợ lý không hợp lệ."}}, status_code=422)
    return JSONResponse({"detail": [{"loc": item['loc'], "msg": item['msg'], "type": item['type']} for item in error.errors()]}, status_code=422)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "routing_profile": "pedestrian", "api_version": "2"}


async def read_upload_image(upload: UploadFile) -> Image.Image:
    image = upload
    max_bytes = int(os.getenv("DETECT_MAX_BYTES", str(DEFAULT_DETECT_MAX_BYTES)))
    chunk_size = max(1, min(65536, max_bytes + 1))
    chunks = bytearray()
    while True:
        chunk = await image.read(chunk_size)
        if not chunk:
            break
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            logger.warning(
                "Upload exceeds size limit: filename=%s content_type=%s limit=%d",
                image.filename,
                image.content_type,
                max_bytes,
            )
            raise HTTPException(status_code=413, detail="Uploaded image exceeds maximum size.")

    data = bytes(chunks)
    if not data:
        logger.warning("Empty upload: filename=%s content_type=%s", image.filename, image.content_type)
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            pil_image = Image.open(io.BytesIO(data))
            if pil_image.width * pil_image.height > DETECT_MAX_PIXELS:
                raise HTTPException(status_code=413, detail="Image exceeds maximum pixel limit.")
            pil_image.load()
            pil_image = ImageOps.exif_transpose(pil_image)
    except HTTPException:
        raise
    except Image.DecompressionBombError as error:
        raise HTTPException(status_code=413, detail="Image exceeds maximum pixel limit.") from error
    except (UnidentifiedImageError, OSError) as error:
        logger.warning(
            "Invalid image upload: filename=%s content_type=%s bytes=%d",
            image.filename,
            image.content_type,
            len(data),
        )
        raise HTTPException(status_code=400, detail="Invalid image data.") from error

    return pil_image


@app.post("/detect")
async def detect(image: UploadFile = File(...), diagnostics: bool = False) -> dict[str, object]:
    pil_image = await read_upload_image(image)
    width, height = pil_image.size
    started = time.perf_counter()
    try:
        objects, source = await run_in_threadpool(run_detection, pil_image)
    except Exception as error:
        logger.exception("YOLO inference failed")
        raise HTTPException(status_code=500, detail="Object detection failed.") from error

    response = {
        "success": True,
        "image_width": width,
        "image_height": height,
        "objects": objects,
    }
    if diagnostics:
        response["diagnostics"] = {
            "model": source["model"],
            "runtime": source["runtime"],
            "source": source["source"],
            # Includes threadpool/lock wait, preprocessing, predict and rules.
            "inference_ms": (time.perf_counter() - started) * 1000,
        }
    return response


@app.post("/segment")
async def segment(image: UploadFile = File(...), diagnostics: bool = False,
                  mask: Literal["none", "png", "rle"] = "none") -> dict[str, object]:
    pil_image = await read_upload_image(image)
    worker = segmentation_worker
    if worker is None:
        raise HTTPException(status_code=503, detail="Segmentation worker unavailable.")
    started = time.perf_counter()
    try:
        result = await run_in_threadpool(worker.segment, pil_image, mask)
    except SegmentationUnavailable as error:
        raise HTTPException(status_code=503, detail="Segmentation worker unavailable.") from error
    except SegmentationBusy as error:
        raise HTTPException(status_code=503, detail="Segmentation worker busy.") from error
    except Exception as error:
        logger.exception("Segmentation inference failed")
        raise HTTPException(status_code=500, detail="Segmentation failed.") from error
    response = {
        "success": True,
        "image_width": pil_image.width,
        "image_height": pil_image.height,
        **result.payload,
        "cached": result.cached,
    }
    if diagnostics:
        response["diagnostics"] = {
            "source": "colab", "model": result.model, "runtime": result.runtime,
            "inference_ms": (time.perf_counter() - started) * 1000,
        }
        if result.cached:
            response["diagnostics"]["cache_age_ms"] = result.cache_age_ms
    return response


@app.get("/geocode")
async def geocode_place(q: str = Query(..., min_length=1, max_length=300),
                        near_lat: float | None = Query(None, ge=-90, le=90),
                        near_lng: float | None = Query(None, ge=-180, le=180)) -> dict[str, object]:
    try:
        place = await geocode(q, near_lat, near_lng)
    except GeocodeServiceError as error:
        logger.warning("Geocode upstream failed for query=%r: %s", q, error)
        raise HTTPException(status_code=502, detail=str(error)) from error
    except GeocodeError as error:
        logger.warning("Geocode failed for query=%r: %s", q, error)
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"success": True, **place}


@app.get("/route")
async def get_route(
    from_lat: float = Query(..., ge=-90, le=90), from_lng: float = Query(..., ge=-180, le=180),
    to_lat: float = Query(..., ge=-90, le=90), to_lng: float = Query(..., ge=-180, le=180)
) -> dict[str, object]:
    try:
        result = await plan_route(from_lat, from_lng, to_lat, to_lng)
    except RouteError as error:
        logger.warning("Routing failed: %s", error)
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"success": True, **result}


@app.get("/reverse")
async def reverse_place(lat: float = Query(..., ge=-90, le=90),
                        lng: float = Query(..., ge=-180, le=180)) -> dict[str, object]:
    try:
        result = await reverse_geocode(lat, lng)
    except GeocodeServiceError as error:
        logger.warning("Reverse geocode upstream failed for (%s, %s): %s", lat, lng, error)
        raise HTTPException(status_code=502, detail=str(error)) from error
    except GeocodeError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"success": True, **result}
