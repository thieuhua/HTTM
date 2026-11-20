import '../config/ai_config.dart';
import '../models/walkable_result.dart';

enum WalkableHaptic { none, medium }

class WalkableHint {
  const WalkableHint(this.text, this.priority, this.haptic);
  final String text;
  final int priority;
  final WalkableHaptic haptic;
}

/// Pure advisory policy. Stairs are deliberately excluded: neither their
/// presence nor their absence supports a reliable stair/safety warning.
class WalkableGuidance {
  final Map<String, int> _hits = {};
  final Map<String, DateTime> _spoken = {};

  void reset({bool cooldowns = true}) {
    _hits.clear();
    if (cooldowns) _spoken.clear();
  }

  WalkableHint? update(
    WalkableResult result,
    DateTime now, {
    bool yoloWarnedRecently = false,
    bool allowOutput = true,
  }) {
    final active = <String>{
      ...result.hazards.where((h) => h != 'stairs_ahead'),
      if (result.suggest == 'stop') 'stop',
      if (!result.pathClear && const ['left', 'right'].contains(result.suggest))
        result.suggest,
    };
    _hits.removeWhere((key, _) => !active.contains(key));
    for (final key in active) {
      _hits[key] = (_hits[key] ?? 0) + 1;
    }
    if (!allowOutput) return null;
    final candidates = <String, WalkableHint>{
      'stop': const WalkableHint(
        'Phía trước không thấy lối đi rõ ràng, hãy dừng lại kiểm tra',
        3,
        WalkableHaptic.medium,
      ),
      if (!yoloWarnedRecently)
        'obstacle_ahead': const WalkableHint(
          'Có thể có vật cản phía trước',
          3,
          WalkableHaptic.medium,
        ),
      'left': const WalkableHint(
        'Lối đi bên trái có vẻ trống hơn',
        2,
        WalkableHaptic.none,
      ),
      'right': const WalkableHint(
        'Lối đi bên phải có vẻ trống hơn',
        2,
        WalkableHaptic.none,
      ),
      'curb_ahead': const WalkableHint(
        'Có thể có bờ vỉa phía trước',
        2,
        WalkableHaptic.none,
      ),
      'road_ahead': const WalkableHint(
        'Phía trước có thể là lòng đường',
        2,
        WalkableHaptic.none,
      ),
      'crosswalk_ahead': const WalkableHint(
        'Có thể có vạch qua đường phía trước',
        2,
        WalkableHaptic.none,
      ),
    };
    for (final entry in candidates.entries) {
      final threshold = const ['left', 'right'].contains(entry.key) ? 3 : 2;
      final last = _spoken[entry.key];
      if ((_hits[entry.key] ?? 0) >= threshold &&
          (last == null || now.difference(last) >= AiConfig.warningCooldown)) {
        _spoken[entry.key] = now;
        return entry.value;
      }
    }
    return null;
  }
}
