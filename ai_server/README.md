# Smart Navigation server v2

FastAPI + YOLO11n + Valhalla **pedestrian** + Nominatim search/reverse.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Python 3.11 recommended. Health: `/health`; interactive API: `/docs`.

Routing: `/route?from_lat=&from_lng=&to_lat=&to_lng=`. Returns `profile=pedestrian`, geometry `[lat,lng]`, shape-indexed maneuvers, distance/time. Uses `VALHALLA_URL` (default `https://valhalla1.openstreetmap.de`); no driving fallback. Public service needs Internet; self-host for sustained production traffic.

Geocoding: `/geocode?q=&near_lat=&near_lng=`; `/reverse?lat=&lng=`. Nominatim requests share a per-process 1.1-second interval and 5-minute bounded cache. Run one worker; multi-worker deployment needs shared throttling/cache.

Detection: `/detect` multipart JPEG; model selected with `YOLO_MODEL`, confidence with `YOLO_CONFIDENCE`. Proximity remains a bounding-box heuristic.

Tests: `.venv\Scripts\python.exe -X utf8 -B -m unittest discover -p "test_*.py" -v`.

See [handoff](../docs/HANDOFF_PEDESTRIAN.md) for contract changes and phone validation.
