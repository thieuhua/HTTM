# Smart Navigation for Visually Impaired People

An Android Flutter app with a live OpenStreetMap, GPS/compass guidance,
Valhalla pedestrian routing, a persisted walked trail and periodic replanning
of the remaining path. A FastAPI/YOLO server also detects selected obstacle
classes from a JPEG roughly every 900 ms, with Vietnamese TTS/haptic warnings.

See [DEMO_SETUP.md](DEMO_SETUP.md) for installation, network configuration,
run commands, pre-demo checks, scenarios, and the MVP's academic limitations.

Current pipeline:

`Camera -> HTTP -> YOLO11n -> bounding-box heuristic -> TTS/haptics`

Walking pipeline:

`Destination confirmation -> Valhalla pedestrian geometry -> GPS path matching -> map + TTS`

The walked GPS trail is kept separately from the remaining route. Routing refreshes
every 30 seconds when moving, and earlier after confirmed route deviation.
See [architecture](docs/ARCHITECTURE.md) and [implementation handoffs](docs/HANDOFF_PEDESTRIAN.md).

The bounding-box heuristic is not depth estimation. The planned production
replacement is monocular depth estimation plus walkable-area reasoning.
