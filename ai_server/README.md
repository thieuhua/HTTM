# Smart Navigation server v2

FastAPI + YOLO11 + Valhalla **pedestrian** + Nominatim search/reverse.

```powershell
cd ai_server
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Python 3.11 or newer is required for the assistant's `asyncio.timeout`. The virtualenv lives at the repository root: create it with `py -3.11 -m venv .venv` there first. Commands above run from `ai_server`. Health: `/health`; interactive API: `/docs`. The model is loaded and warmed up at startup (`YOLO_PRELOAD=0` disables that). Behind a tunnel or reverse proxy, its origin port must match the uvicorn port, otherwise the public URL answers 502.

Routing: `/route?from_lat=&from_lng=&to_lat=&to_lng=`. Returns `profile=pedestrian`, geometry `[lat,lng]`, shape-indexed maneuvers, distance/time. Uses `VALHALLA_URL` (default `https://valhalla1.openstreetmap.de`); no driving fallback. Public service needs Internet; self-host for sustained production traffic.

Geocoding: `/geocode?q=&near_lat=&near_lng=`; `/reverse?lat=&lng=`. Nominatim requests share a per-process 1.1-second interval and 5-minute bounded cache. Run one worker; multi-worker deployment needs shared throttling/cache.

Detection: `/detect` multipart JPEG; model selected with `YOLO_MODEL`, confidence with `YOLO_CONFIDENCE`. Proximity remains a bounding-box heuristic. To use the custom trained 10-class model (local weights, not committed to git), set `YOLO_MODEL=httm_yolo11s_v2.pt` in the environment or `ai_server/.env` (v2 = fine-tuned 40 epochs from the 10-epoch baseline; test mAP50 0.873, mAP50-95 0.643) (classes: `Bang hieu`, `Cay coi`, `Cot`, `Den giao thong`, `Nguoi`, `O ga`, `Thung rac`, `Xe dap`, `Xe hoi`, `Xe may`).

`/detect` rejects uploads over `DETECT_MAX_BYTES` (default 5 MB) or 12 MP with 413. `/geocode` and `/reverse` return 404 for "no result" and 502 when Nominatim fails; set `NOMINATIM_CONTACT` for the User-Agent.

Optional walkable segmentation runs alongside YOLO: set `COLAB_SEG_URL=http://127.0.0.1:9001`
to reach a separate Colab `worker_seg` through a loopback SSH tunnel. Only HTTP loopback
addresses are accepted; an empty value disables segmentation. `COLAB_SEG_URL` and
`SEG_MIN_INTERVAL_MS` (default `500`) use process environment first, then `ai_server/.env`.
The worker shares one keep-alive HTTP client with its 4-second health poll, with a
3-second timeout (connect 0.3 s, pool 0.1 s) and 30-second keep-alive expiry.
There is no local segmentation fallback. `/detect` and `/health` keep their existing behavior.

`POST /segment` takes multipart field `image`, with the same upload limits, EXIF
transpose, 400 details and 413 details as `/detect`. Query parameters are
`diagnostics=false` and `mask=none|png|rle` (default `none`; invalid values return 422).
The 200 body contains `success=true`, `image_width`/`image_height` for **this upload
after EXIF transpose**, `classes`, `ratios`, `grid`, `path_clear`, `suggest`, `hazards`,
and `cached`. The ordered classes are `other floor sidewalk road crosswalk stairs
curb terrain obstacle`; `ratios` maps each name to a fraction. `grid` is 3x3:
row 0 is far/top, row 2 near/bottom, columns left/center/right; cells contain
`walkable`, `caution`, `blocked` fractions in [0,1] and a `dominant` class name.
`path_clear` is boolean, `suggest` is `straight|left|right|stop`, and `hazards` is a
subset of `stairs_ahead`, `curb_ahead`, `road_ahead`, `crosswalk_ahead`, `obstacle_ahead`.

Mask modes forward `mask_w`, `mask_h` and `mask_png_b64` or `mask_rle` verbatim.
Masks describe the resized worker image (long side at most 640), not the original
upload dimensions. PNG pixels are class IDs 0..8; RLE is multiclass row-major
`{"size":[h,w],"order":"C","values":[...],"counts":[...]}`, not COCO RLE.
Diagnostics appear only when requested:
`{"source":"colab","model":"best","runtime":"cuda/fp16","inference_ms":123.4}`;
`inference_ms` measures the server call including threadpool wait and preprocessing.
Cached diagnostics also include `cache_age_ms`.

Only one segmentation request can be in flight; additional requests fail immediately.
Requests before `SEG_MIN_INTERVAL_MS` elapses receive the last result for the same
mask mode only if it is less than 1.5 seconds old, otherwise they receive busy.
Cached results may describe a previous camera frame; images are never cached by
content or written to disk. Cache is cleared whenever the worker becomes unavailable.
Errors use fixed JSON details: 503 `Segmentation worker unavailable.` for off/unready/
cooldown/worker 503, 503 `Segmentation worker busy.` for in-flight/interval/pool
contention, and 500 `Segmentation failed.` for unexpected failures. Worker 503
starts a 5-second cooldown; HTTP/connection/timeout/malformed reply failures start
a 10-second cooldown, followed by health-based recovery. Pool timeout keeps readiness.

Use **only model v2, `runs/segformer_b5_v2/best`, never v1**. Health reports just
the checkpoint directory name (`best`), so confirm the full v2 path in Colab logs;
the server cannot distinguish v1 from v2 by health alone. The `stairs` class is weak
(IoU about 0.18) and **must not be used for stair warnings**, including inference
from an absent `stairs_ahead`. Curb/crosswalk are hints only. Summaries are image-space
heuristics, not depth, distance, safe crossing, or autonomous navigation.
See [walkable handoff](../docs/HANDOFF_WALKABLE_SEG.md) for worker/tunnel setup.

Tests: `..\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -p "test_*.py" -v`.

Segmentation tests use `httpx.MockTransport`, injected clocks, and the real worker
JSON fixture in `fixtures/walkable_worker_reply.json`, generated offline with
`worker_seg.create_app` and a fake model in the read-only walkable-seg checkout.
No socket, GPU model, or live tunnel is needed for these tests.

See [handoff](../docs/HANDOFF_PEDESTRIAN.md) for contract changes and phone validation.

Optional AI Assistant (OpenAI or Gemini): `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ASSISTANT_PROVIDER` (`openai` or `gemini`), `OPENAI_ASSISTANT_MODEL`, and `GEMINI_ASSISTANT_MODEL` are read only from this directory's `.env`.
For a new clone, copy `.env.example` to `.env` only if it does not exist, then set the key locally. Never copy secrets from the reference fork or into Flutter assets.
Run `python setup_assistant.py` to provision signing/pairing secrets; use
`--show-pairing-code` to give the device its pairing code, never the OpenAI or Gemini key.
Assistant endpoints require HTTPS, device bearer auth, and persistent SQLite
rate/spend limits. No tool directly changes navigation. Defaults: 1 USD/day total,
0.5 USD/day/device, 6 turns/min/device, 0.05 USD maximum reservation/turn.
See [OpenAI handoff](../docs/HANDOFF_OPENAI_ASSISTANT.md) for configuration and local
HTTP debug exceptions. Navigation endpoints remain independent of assistant configuration.

Run `python setup_assistant.py --qr` to create `.assistant-state/pairing-qr.png`
locally. Keep it private: it contains the device pairing credential, not the OpenAI
key or signing secret. It is ignored by Git and is not served by the backend.
Scan it in the app's pairing screen, then press Pair. Existing secrets are preserved.
