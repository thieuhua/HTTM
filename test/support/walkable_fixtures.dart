import 'package:smart_navigation/models/walkable_result.dart';

Map<String, dynamic> walkableJson({
  String suggest = 'straight',
  bool pathClear = true,
  List<String> hazards = const [],
}) => {
  'success': true,
  'image_width': 100,
  'image_height': 80,
  'classes': [...WalkableResult.classes],
  'ratios': {
    for (final name in WalkableResult.classes)
      name: name == 'floor' ? 1.0 : 0.0,
  },
  'grid': List.generate(
    3,
    (_) => List.generate(
      3,
      (_) => <String, dynamic>{
        'walkable': 1.0,
        'caution': 0.0,
        'blocked': 0.0,
        'dominant': 'floor',
      },
    ),
  ),
  'path_clear': pathClear,
  'suggest': suggest,
  'hazards': hazards,
  'cached': false,
};

WalkableResult walkable({
  String suggest = 'straight',
  bool pathClear = true,
  List<String> hazards = const [],
}) => WalkableResult.fromJson(
  walkableJson(suggest: suggest, pathClear: pathClear, hazards: hazards),
);
