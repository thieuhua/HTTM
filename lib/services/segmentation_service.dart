import 'dart:async';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import '../config/ai_config.dart';
import '../models/walkable_result.dart';

class SegmentationException implements Exception {
  const SegmentationException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Both worker-unavailable and worker-busy are optional perception outages.
class SegmentationUnavailableException extends SegmentationException {
  const SegmentationUnavailableException() : super('Phân vùng lối đi tạm tắt');
}

/// The server is still answering the previous frame; skip quietly.
class SegmentationBusyException extends SegmentationUnavailableException {
  const SegmentationBusyException();
}

class SegmentationService {
  SegmentationService({
    http.Client? client,
    Duration timeout = AiConfig.segmentTimeout,
  }) : assert(timeout > Duration.zero),
       _client = client ?? http.Client(),
       _timeout = timeout;

  final http.Client _client;
  final Duration _timeout;
  final Set<Completer<void>> _pending = {};
  bool _disposed = false;

  Future<WalkableResult> segment(XFile image) async {
    if (_disposed) throw const SegmentationException('Segmentation disposed.');
    final abort = Completer<void>();
    _pending.add(abort);
    try {
      // One deadline includes image preparation, upload, headers and body.
      return await Future.any<WalkableResult>([
        _segment(image, abort),
        abort.future.then<WalkableResult>(
          (_) => throw const SegmentationException('Segmentation cancelled.'),
        ),
      ]).timeout(
        _timeout,
        onTimeout: () {
          if (!abort.isCompleted) abort.complete();
          throw const SegmentationException('Segmentation timed out.');
        },
      );
    } on SegmentationException {
      rethrow;
    } catch (_) {
      throw const SegmentationException('Segmentation request failed.');
    } finally {
      _pending.remove(abort);
    }
  }

  Future<WalkableResult> _segment(XFile image, Completer<void> abort) async {
    final request = http.AbortableMultipartRequest(
      'POST',
      Uri.parse('${AiConfig.serverUrl}/segment?mask=none'),
      abortTrigger: abort.future,
    );
    request.files.add(
      http.MultipartFile.fromBytes(
        'image',
        await image.readAsBytes(),
        filename: 'frame.jpg',
        contentType: MediaType('image', 'jpeg'),
      ),
    );
    if (abort.isCompleted) throw http.RequestAbortedException(request.url);
    final response = await http.Response.fromStream(
      await _client.send(request),
    );
    if (response.statusCode == 503) {
      if (_detail(response.body) == 'Segmentation worker busy.') {
        throw const SegmentationBusyException();
      }
      throw const SegmentationUnavailableException();
    }
    if (response.statusCode != 200) {
      throw SegmentationException(
        'Segmentation failed (${response.statusCode}).',
      );
    }
    final body = jsonDecode(response.body);
    if (body is! Map<String, dynamic>) {
      throw const FormatException('Invalid segmentation response.');
    }
    return WalkableResult.fromJson(body);
  }

  static Object? _detail(String body) {
    try {
      final decoded = jsonDecode(body);
      return decoded is Map ? decoded['detail'] : null;
    } catch (_) {
      return null;
    }
  }

  void cancelPending() {
    for (final abort in _pending) {
      if (!abort.isCompleted) abort.complete();
    }
  }

  void dispose() {
    _disposed = true;
    cancelPending();
    _client.close();
  }
}
