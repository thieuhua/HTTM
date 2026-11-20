import 'package:flutter_test/flutter_test.dart';
import 'package:smart_navigation/config/ai_config.dart';
import 'package:smart_navigation/services/walkable_guidance.dart';

import 'support/walkable_fixtures.dart';

void main() {
  final now = DateTime.utc(2026, 11, 20);
  test(
    'straight and stairs presence/absence never announce safety or stairs',
    () {
      final guidance = WalkableGuidance();
      for (var i = 0; i < 10; i++) {
        expect(
          guidance.update(walkable(hazards: ['stairs_ahead']), now),
          isNull,
        );
        expect(guidance.update(walkable(), now), isNull);
      }
    },
  );
  for (final side in ['left', 'right']) {
    test(
      '$side needs three consecutive unclear results, changes reset hysteresis',
      () {
        final guidance = WalkableGuidance();
        final result = walkable(suggest: side, pathClear: false);
        expect(guidance.update(result, now), isNull);
        expect(guidance.update(result, now), isNull);
        guidance.update(walkable(), now);
        expect(guidance.update(result, now), isNull);
        expect(guidance.update(result, now), isNull);
        final hint = guidance.update(result, now)!;
        expect(
          hint.text,
          'Lối đi bên ${side == 'left' ? 'trái' : 'phải'} có vẻ trống hơn',
        );
        expect(hint.priority, 2);
        expect(hint.haptic, WalkableHaptic.none);
        expect(
          guidance.update(result, now.add(const Duration(seconds: 1))),
          isNull,
        );
        expect(
          guidance.update(result, now.add(AiConfig.warningCooldown)),
          isNotNull,
        );
      },
    );
  }
  test('path_clear true blocks side suggestion even if suggest is left', () {
    final guidance = WalkableGuidance();
    for (var i = 0; i < 4; i++) {
      expect(guidance.update(walkable(suggest: 'left'), now), isNull);
    }
  });
  test(
    'stop after two results gives cautious text, priority 3, medium haptic',
    () {
      final guidance = WalkableGuidance();
      final stop = walkable(suggest: 'stop', pathClear: false);
      expect(guidance.update(stop, now), isNull);
      final hint = guidance.update(stop, now)!;
      expect(
        hint.text,
        'Phía trước không thấy lối đi rõ ràng, hãy dừng lại kiểm tra',
      );
      expect(hint.priority, 3);
      expect(hint.haptic, WalkableHaptic.medium);
    },
  );
  final hints = {
    'curb_ahead': 'Có thể có bờ vỉa phía trước',
    'road_ahead': 'Phía trước có thể là lòng đường',
    'crosswalk_ahead': 'Có thể có vạch qua đường phía trước',
  };
  for (final entry in hints.entries) {
    test('${entry.key} requires consecutive hits and uses hedged wording', () {
      final guidance = WalkableGuidance();
      final result = walkable(hazards: [entry.key]);
      expect(guidance.update(result, now), isNull);
      guidance.update(walkable(), now);
      expect(guidance.update(result, now), isNull);
      final hint = guidance.update(result, now)!;
      expect(hint.text, entry.value);
      expect(hint.priority, 2);
      expect(hint.haptic, WalkableHaptic.none);
    });
  }
  test(
    'obstacle suppressed for recent YOLO warning, cooldowns are per key',
    () {
      final guidance = WalkableGuidance();
      final obstacle = walkable(hazards: ['obstacle_ahead']);
      guidance.update(obstacle, now, yoloWarnedRecently: true);
      expect(guidance.update(obstacle, now, yoloWarnedRecently: true), isNull);
      expect(
        guidance.update(obstacle, now)?.text,
        'Có thể có vật cản phía trước',
      );
      final road = walkable(hazards: ['road_ahead']);
      expect(guidance.update(road, now), isNull);
      expect(guidance.update(road, now)?.text, hints['road_ahead']);
    },
  );
  test('speech holds accumulate hits without consuming per-key cooldown', () {
    final guidance = WalkableGuidance();
    final stop = walkable(suggest: 'stop');
    guidance.update(stop, now, allowOutput: false);
    expect(guidance.update(stop, now, allowOutput: false), isNull);
    expect(guidance.update(stop, now), isNotNull);
    guidance.reset(cooldowns: false);
    guidance.update(stop, now);
    expect(guidance.update(stop, now), isNull);
    guidance.reset();
    guidance.update(stop, now);
    expect(guidance.update(stop, now), isNotNull);
  });
}
