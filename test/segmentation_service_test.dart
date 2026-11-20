import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:smart_navigation/config/ai_config.dart';
import 'package:smart_navigation/models/walkable_result.dart';
import 'package:smart_navigation/services/segmentation_service.dart';

import 'support/walkable_fixtures.dart';

class SegmentClient extends http.BaseClient {
  SegmentClient(this.reply);
  final Future<http.StreamedResponse> Function(http.AbortableMultipartRequest)
  reply;
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    await request.finalize().drain<void>();
    return reply(request as http.AbortableMultipartRequest);
  }
}

http.StreamedResponse reply([Object? body, int status = 200]) =>
    http.StreamedResponse(
      Stream.value(utf8.encode(jsonEncode(body ?? walkableJson()))),
      status,
    );

void main() {
  final image = XFile.fromData(Uint8List.fromList([1, 2, 3]));
  setUp(
    () => dotenv.loadFromString(
      envString: 'AI_SERVER_URL=https://backend.example',
    ),
  );

  test('flag defaults off, dotenv enables explicitly, timings are bounded', () {
    expect(AiConfig.walkableSegEnabled, isFalse);
    dotenv.loadFromString(envString: 'WALKABLE_SEG=true');
    expect(AiConfig.walkableSegEnabled, isTrue);
    dotenv.loadFromString(envString: 'WALKABLE_SEG=invalid');
    expect(AiConfig.walkableSegEnabled, isFalse);
    expect(AiConfig.segmentTimeout, const Duration(seconds: 4));
    expect(AiConfig.segmentInterval, const Duration(milliseconds: 1800));
  });

  test(
    'same bytes uploaded as multipart image with no mask/health request',
    () async {
      final service = SegmentationService(
        client: SegmentClient((request) async {
          expect(request.method, 'POST');
          expect(
            request.url.toString(),
            'https://backend.example/segment?mask=none',
          );
          expect(request.files.single.field, 'image');
          expect(request.files.single.length, 3);
          expect(request.files.single.contentType.toString(), 'image/jpeg');
          return reply();
        }),
      );
      addTearDown(service.dispose);
      final result = await service.segment(image);
      expect(result.imageWidth, 100);
      expect(result.imageHeight, 80);
      expect(result.grid[2][1].dominant, 'floor');
      expect(result.ratios['floor'], 1);
      expect(result.cached, isFalse);
    },
  );

  for (final detail in [
    'Segmentation worker unavailable.',
    'Segmentation worker busy.',
  ]) {
    test('503 $detail is a distinct nonfatal outage', () async {
      final service = SegmentationService(
        client: SegmentClient((_) async => reply({'detail': detail}, 503)),
      );
      addTearDown(service.dispose);
      await expectLater(
        service.segment(image),
        throwsA(isA<SegmentationUnavailableException>()),
      );
    });
  }
  test('only the busy detail maps to SegmentationBusyException', () async {
    final busy = SegmentationService(
      client: SegmentClient(
        (_) async => reply({'detail': 'Segmentation worker busy.'}, 503),
      ),
    );
    final off = SegmentationService(
      client: SegmentClient(
        (_) async => reply({'detail': 'Segmentation worker unavailable.'}, 503),
      ),
    );
    addTearDown(busy.dispose);
    addTearDown(off.dispose);
    await expectLater(
      busy.segment(image),
      throwsA(isA<SegmentationBusyException>()),
    );
    await expectLater(
      off.segment(image),
      throwsA(
        allOf(
          isA<SegmentationUnavailableException>(),
          isNot(isA<SegmentationBusyException>()),
        ),
      ),
    );
  });
  for (final status in [400, 413, 422, 500]) {
    test('HTTP $status produces generic segmentation exception', () async {
      final service = SegmentationService(
        client: SegmentClient((_) async => reply({}, status)),
      );
      addTearDown(service.dispose);
      await expectLater(
        service.segment(image),
        throwsA(
          isA<SegmentationException>()
              .having(
                (e) => e is SegmentationUnavailableException,
                'outage',
                false,
              )
              .having((e) => e.message, 'status', contains('$status')),
        ),
      );
    });
  }

  for (final body in ['broken JSON', '[]', '{}']) {
    test('malformed/missing summary $body is a generic error', () async {
      final service = SegmentationService(
        client: SegmentClient(
          (_) async =>
              http.StreamedResponse(Stream.value(utf8.encode(body)), 200),
        ),
      );
      addTearDown(service.dispose);
      await expectLater(
        service.segment(image),
        throwsA(isA<SegmentationException>()),
      );
    });
  }

  final invalid = <String, void Function(Map<String, dynamic>)>{
    'success': (j) => j['success'] = false,
    'missing width': (j) => j.remove('image_width'),
    'fractional width': (j) => j['image_width'] = 1.5,
    'zero height': (j) => j['image_height'] = 0,
    'classes order': (j) =>
        j['classes'] = WalkableResult.classes.reversed.toList(),
    'missing ratio': (j) => (j['ratios'] as Map).remove('floor'),
    'NaN ratio': (j) => j['ratios']['floor'] = double.nan,
    'infinite ratio': (j) => j['ratios']['floor'] = double.infinity,
    'negative ratio': (j) => j['ratios']['floor'] = -.1,
    'large ratio': (j) => j['ratios']['floor'] = 1.1,
    'grid rows': (j) => j['grid'] = [],
    'grid columns': (j) => j['grid'][0] = [],
    'grid cell': (j) => j['grid'][0][0] = null,
    'grid probability': (j) => j['grid'][2][1]['walkable'] = '1',
    'grid dominant': (j) => j['grid'][2][1]['dominant'] = 'unknown',
    'path_clear': (j) => j['path_clear'] = 1,
    'suggest': (j) => j['suggest'] = 'safe',
    'hazard': (j) => j['hazards'] = ['safe'],
    'cached': (j) => j.remove('cached'),
    'mask': (j) => j['mask_png_b64'] = 'png',
    'diagnostics': (j) => j['diagnostics'] = [],
  };
  for (final entry in invalid.entries) {
    test('strict fromJson rejects ${entry.key} with FormatException', () {
      // Round-trip so nested lists/maps are loosely typed like real JSON.
      final json =
          jsonDecode(jsonEncode(walkableJson())) as Map<String, dynamic>;
      entry.value(json);
      expect(() => WalkableResult.fromJson(json), throwsFormatException);
    });
  }

  test('optional masks and cached diagnostics survive parsing verbatim', () {
    final json = walkableJson()
      ..addAll({
        'cached': true,
        'mask_w': 2,
        'mask_h': 1,
        'mask_png_b64': 'png',
        'diagnostics': {
          'source': 'colab',
          'model': 'best',
          'runtime': 'cuda/fp16',
          'inference_ms': 125.5,
          'cache_age_ms': 80.0,
        },
      });
    final png = WalkableResult.fromJson(json);
    expect(png.maskPngBase64, 'png');
    expect(png.diagnostics, json['diagnostics']);
    json.remove('mask_png_b64');
    json['mask_rle'] = {
      'size': [1, 2],
      'order': 'C',
      'values': [1],
      'counts': [2],
    };
    expect(WalkableResult.fromJson(json).maskRle, json['mask_rle']);
  });

  test('stalled headers abort, and subsequent request recovers', () async {
    var first = true;
    final aborted = Completer<void>();
    final service = SegmentationService(
      timeout: const Duration(milliseconds: 80),
      client: SegmentClient((request) async {
        if (!first) return reply();
        first = false;
        await request.abortTrigger;
        aborted.complete();
        throw http.RequestAbortedException(request.url);
      }),
    );
    addTearDown(service.dispose);
    await expectLater(
      service.segment(image),
      throwsA(isA<SegmentationException>()),
    );
    await aborted.future;
    expect((await service.segment(image)).pathClear, isTrue);
  });

  test('headers and stalled body share a single deadline and abort', () async {
    final body = StreamController<List<int>>();
    final service = SegmentationService(
      timeout: const Duration(milliseconds: 120),
      client: SegmentClient((request) async {
        unawaited(
          request.abortTrigger!.then((_) {
            body.addError(http.RequestAbortedException(request.url));
            unawaited(body.close());
          }),
        );
        await Future<void>.delayed(const Duration(milliseconds: 80));
        unawaited(
          Future<void>.delayed(const Duration(milliseconds: 80), () {
            if (!body.isClosed) {
              body.add(utf8.encode(jsonEncode(walkableJson())));
              unawaited(body.close());
            }
          }),
        );
        return http.StreamedResponse(body.stream, 200);
      }),
    );
    addTearDown(service.dispose);
    await expectLater(
      service.segment(image),
      throwsA(isA<SegmentationException>()),
    );
    await body.done;
  });

  test(
    'cancel and dispose abort pending requests without waiting for deadline',
    () async {
      final entered = Completer<void>(), aborted = Completer<void>();
      final service = SegmentationService(
        client: SegmentClient((request) async {
          entered.complete();
          await request.abortTrigger;
          aborted.complete();
          throw http.RequestAbortedException(request.url);
        }),
      );
      final future = service.segment(image);
      final assertion = expectLater(
        future,
        throwsA(isA<SegmentationException>()),
      );
      await entered.future;
      service.cancelPending();
      await assertion;
      await aborted.future;
      service.dispose();
      await expectLater(
        service.segment(image),
        throwsA(isA<SegmentationException>()),
      );
    },
  );
}
