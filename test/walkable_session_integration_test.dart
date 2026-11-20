import 'dart:async';

import 'package:camera/camera.dart';
// ignore: depend_on_referenced_packages
import 'package:fake_async/fake_async.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:smart_navigation/config/ai_config.dart';
import 'package:smart_navigation/controllers/session_controller.dart';
import 'package:smart_navigation/models/perception_frame.dart';
import 'package:smart_navigation/models/walkable_result.dart';
import 'package:smart_navigation/services/segmentation_service.dart';

import 'assistant_test.dart' show FakeAssistant;
import 'segmentation_service_test.dart' show SegmentClient;
import 'session_navigation_test.dart' show FakeRoute;
import 'support/device_fakes.dart';
import 'support/walkable_fixtures.dart';

class FakeSegmentation extends SegmentationService {
  int requests = 0, cancellations = 0;
  bool fail = false, busy = false, disposed = false, deferred = false;
  bool serverBusy = false;
  WalkableResult result = walkable(suggest: 'stop', pathClear: false);
  final pending = <Completer<WalkableResult>>[];
  final images = <XFile>[];
  @override
  Future<WalkableResult> segment(XFile image) async {
    requests++;
    images.add(image);
    if (serverBusy) throw const SegmentationBusyException();
    if (busy) throw const SegmentationUnavailableException();
    if (fail) throw const SegmentationException('500');
    if (!deferred) return result;
    final response = Completer<WalkableResult>();
    pending.add(response);
    return response.future;
  }

  @override
  void cancelPending() {
    cancellations++;
    // Deliberately allow late replies: epoch checks must reject them even
    // when a backend fails to honor cancellation immediately.
  }

  @override
  void dispose() {
    disposed = true;
    super.dispose();
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final haptics = <String>[];
  setUp(() {
    dotenv.loadFromString(envString: 'WALKABLE_SEG=true');
    haptics.clear();
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(SystemChannels.platform, (call) async {
          if (call.method == 'HapticFeedback.vibrate') {
            haptics.add(call.arguments as String);
          }
          return null;
        });
  });
  tearDown(() {
    dotenv.loadFromString(envString: '', isOptional: true);
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(SystemChannels.platform, null);
  });

  SessionController session(
    FakeAsync async,
    FakeCamera camera,
    FakeDetector detector,
    FakeSegmentation seg,
    RecordingTts tts,
  ) => SessionController(
    frameSource: camera,
    detectionService: detector,
    segmentationService: seg,
    ttsService: tts,
    voiceService: FakeVoice(),
    routeService: FakeRoute(),
    assistantService: FakeAssistant(),
    now: async.getClock(DateTime.utc(2026, 11, 20)).now,
  );

  test(
    'default OFF never invokes seg; cadence ON reuses capture and does not block YOLO',
    () {
      fakeAsync((async) {
        dotenv.loadFromString(envString: '', isOptional: true);
        final camera = FakeCamera(),
            detector = FakeDetector(),
            seg = FakeSegmentation(),
            tts = RecordingTts();
        final controller = session(async, camera, detector, seg, tts);
        controller.initialize();
        async.flushMicrotasks();
        controller.start();
        async.flushMicrotasks();
        async.elapse(AiConfig.segmentInterval * 2);
        expect(seg.requests, 0);
        expect(detector.requests, camera.captures);
        dotenv.loadFromString(envString: 'WALKABLE_SEG=true');
        seg.deferred = true;
        async.elapse(AiConfig.captureInterval);
        expect(seg.requests, 1);
        final yoloBefore = detector.requests;
        async.elapse(AiConfig.segmentInterval * 2);
        expect(seg.requests, 1); // Own in-progress guard.
        expect(detector.requests, greaterThan(yoloBefore));
        expect(camera.captures, detector.requests); // No second capture.
        var sameBytes = false;
        seg.images.single.readAsBytes().then((bytes) {
          sameBytes = bytes.toString() == '[1, 2, 3]';
        });
        async.flushMicrotasks();
        expect(sameBytes, isTrue);
        seg.pending.single.complete(walkable());
        async.flushMicrotasks();
        seg.deferred = false;
        async.elapse(AiConfig.segmentInterval * 2);
        expect(seg.requests, lessThan(detector.requests));
        controller.dispose();
        async.flushMicrotasks();
      });
    },
  );

  for (final busy in [true, false]) {
    test(
      '${busy ? '503' : 'generic error'} backs off once per outage; YOLO and GPS continue',
      () {
        fakeAsync((async) {
          final camera = FakeCamera(),
              detector = FakeDetector()..result = dangerDetection,
              seg = FakeSegmentation()
                ..busy = busy
                ..fail = !busy,
              tts = RecordingTts(),
              route = FakeRoute();
          final controller = SessionController(
            frameSource: camera,
            detectionService: detector,
            segmentationService: seg,
            ttsService: tts,
            voiceService: FakeVoice(),
            routeService: route,
            assistantService: FakeAssistant(),
            now: async.getClock(DateTime.utc(2026, 11, 20)).now,
          );
          final frames = <PerceptionFrame>[];
          final subscription = controller.frames.listen(frames.add);
          controller.initialize();
          async.flushMicrotasks();
          controller.start();
          async.flushMicrotasks();
          expect(seg.requests, 1);
          async.elapse(const Duration(seconds: 9));
          expect(seg.requests, 1);
          async.elapse(const Duration(seconds: 14));
          expect(seg.requests, 3);
          expect(
            tts.spoken.where((s) => s.$1 == 'Phân vùng lối đi tạm tắt').length,
            lessThanOrEqualTo(1),
          );
          expect(
            tts.spoken.any((s) => s.$1.contains('đang thiếu dữ liệu')),
            isFalse,
          );
          expect(tts.spoken.any((s) => s.$2 == 3), isTrue);
          expect(frames.last.objects.single.isStable, isTrue);
          expect(
            frames.last.objects.single.trackId,
            frames.first.objects.single.trackId,
          );
          expect(controller.cameraError, isNull);
          expect(controller.isRunning, isTrue);
          expect(route.started, 1);
          expect(route.stopped, 0);
          subscription.cancel();
          controller.dispose();
          async.flushMicrotasks();
        });
      },
    );
  }

  test(
    'outage announcement resets after recovery and shares obstacle speech gap',
    () {
      fakeAsync((async) {
        final camera = FakeCamera(),
            detector = FakeDetector(),
            seg = FakeSegmentation()..busy = true,
            tts = RecordingTts();
        final controller = session(async, camera, detector, seg, tts);
        controller.initialize();
        async.flushMicrotasks();
        controller.start();
        async.flushMicrotasks();
        int outages() =>
            tts.spoken.where((s) => s.$1 == 'Phân vùng lối đi tạm tắt').length;
        expect(outages(), 1);
        async.elapse(const Duration(seconds: 11));
        expect(outages(), 1);
        seg.busy = false;
        seg.result = walkable();
        async.elapse(const Duration(seconds: 11));
        seg.busy = true;
        async.elapse(AiConfig.segmentInterval);
        expect(outages(), 2);
        controller.dispose();
        async.flushMicrotasks();
      });
    },
  );

  test('server busy is skipped silently without outage backoff', () {
    fakeAsync((async) {
      final camera = FakeCamera(),
          detector = FakeDetector(),
          seg = FakeSegmentation()..serverBusy = true,
          tts = RecordingTts();
      final controller = session(async, camera, detector, seg, tts);
      controller.initialize();
      async.flushMicrotasks();
      controller.start();
      async.flushMicrotasks();
      final first = seg.requests;
      expect(first, greaterThan(0));
      // No 10 s backoff: the next cadence slot sends again.
      async.elapse(AiConfig.segmentInterval * 2);
      expect(seg.requests, greaterThan(first));
      expect(
        tts.spoken.where((s) => s.$1 == 'Phân vùng lối đi tạm tắt'),
        isEmpty,
      );
      controller.dispose();
      async.flushMicrotasks();
    });
  });

  test(
    'two stable stop replies speak once; shared gap suppresses further seg speech',
    () {
      fakeAsync((async) {
        final camera = FakeCamera(),
            detector = FakeDetector(),
            seg = FakeSegmentation(),
            tts = RecordingTts();
        final controller = session(async, camera, detector, seg, tts);
        controller.initialize();
        async.flushMicrotasks();
        controller.start();
        async.flushMicrotasks();
        expect(
          tts.spoken.where((s) => s.$1.startsWith('Phía trước không thấy')),
          isEmpty,
        );
        async.elapse(AiConfig.segmentInterval);
        expect(
          tts.spoken
              .where((s) => s.$1.startsWith('Phía trước không thấy'))
              .single
              .$2,
          3,
        );
        expect(haptics, contains('HapticFeedbackType.mediumImpact'));
        seg.result = walkable(hazards: ['road_ahead']);
        async.elapse(AiConfig.segmentInterval * 2);
        expect(
          tts.spoken.any((s) => s.$1 == 'Phía trước có thể là lòng đường'),
          isFalse,
        );
        controller.dispose();
        async.flushMicrotasks();
      });
    },
  );

  test(
    'results older than detection TTL reset hysteresis and are discarded',
    () {
      fakeAsync((async) {
        final camera = FakeCamera(),
            detector = FakeDetector(),
            seg = FakeSegmentation(),
            tts = RecordingTts();
        final controller = session(async, camera, detector, seg, tts);
        controller.initialize();
        async.flushMicrotasks();
        controller.start();
        async.flushMicrotasks(); // First stop observation.
        seg.deferred = true;
        async.elapse(AiConfig.segmentInterval);
        async.elapse(const Duration(seconds: 6));
        seg.pending.single.complete(seg.result);
        async.flushMicrotasks();
        expect(
          tts.spoken.any((s) => s.$1.startsWith('Phía trước không thấy')),
          isFalse,
        );
        seg.deferred = false;
        async.elapse(AiConfig.captureInterval);
        expect(
          tts.spoken.any((s) => s.$1.startsWith('Phía trước không thấy')),
          isFalse,
        );
        async.elapse(AiConfig.segmentInterval);
        expect(
          tts.spoken.any((s) => s.$1.startsWith('Phía trước không thấy')),
          isTrue,
        );
        controller.dispose();
        async.flushMicrotasks();
      });
    },
  );

  for (final disposing in [false, true]) {
    test(
      '${disposing ? 'dispose' : 'stop'} aborts actual multipart request',
      () {
        fakeAsync((async) {
          var uploaded = false, aborted = false;
          final segmentation = SegmentationService(
            client: SegmentClient((request) async {
              uploaded = true;
              await request.abortTrigger;
              aborted = true;
              throw const SegmentationException('Cancelled');
            }),
          );
          final detector = FakeDetector(), camera = FakeCamera();
          final controller = SessionController(
            frameSource: camera,
            detectionService: detector,
            segmentationService: segmentation,
            ttsService: RecordingTts(),
            voiceService: FakeVoice(),
            routeService: FakeRoute(),
            assistantService: FakeAssistant(),
            now: async.getClock(DateTime.utc(2026, 11, 20)).now,
          );
          controller.initialize();
          async.flushMicrotasks();
          controller.start();
          async.flushMicrotasks();
          expect(uploaded, isTrue);
          async.elapse(AiConfig.captureInterval);
          expect(detector.requests, 2);
          if (disposing) {
            controller.dispose();
          } else {
            controller.stop();
          }
          async.flushMicrotasks();
          expect(aborted, isTrue);
          if (!disposing) {
            controller.dispose();
            async.flushMicrotasks();
          }
        });
      },
    );

    test(
      '${disposing ? 'dispose' : 'stop/restart'} cancels in-flight seg and ignores late reply',
      () {
        fakeAsync((async) {
          final camera = FakeCamera(),
              detector = FakeDetector(),
              seg = FakeSegmentation(),
              tts = RecordingTts();
          final controller = session(async, camera, detector, seg, tts);
          controller.initialize();
          async.flushMicrotasks();
          controller.start();
          async.flushMicrotasks();
          seg.deferred = true;
          async.elapse(AiConfig.segmentInterval);
          if (disposing) {
            controller.dispose();
          } else {
            controller.stop();
          }
          async.flushMicrotasks();
          expect(seg.cancellations, greaterThan(0));
          if (!disposing) {
            controller.start();
            async.flushMicrotasks();
          }
          seg.pending.single.complete(seg.result);
          async.flushMicrotasks();
          expect(
            tts.spoken.any((s) => s.$1.startsWith('Phía trước không thấy')),
            isFalse,
          );
          if (!disposing) {
            controller.dispose();
            async.flushMicrotasks();
          }
          // dispose() awaits camera/voice teardown before releasing seg.
          async.elapse(const Duration(seconds: 5));
          expect(seg.disposed, isTrue);
        });
      },
    );
  }
}
