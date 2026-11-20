/// The summary returned by ai_server /segment. No safety claim is inferred
/// from an absent hazard or from pathClear.
class WalkableResult {
  WalkableResult._({
    required this.imageWidth,
    required this.imageHeight,
    required this.ratios,
    required this.grid,
    required this.pathClear,
    required this.suggest,
    required this.hazards,
    required this.cached,
    this.maskWidth,
    this.maskHeight,
    this.maskPngBase64,
    this.maskRle,
    this.diagnostics,
  });

  static const classes = [
    'other',
    'floor',
    'sidewalk',
    'road',
    'crosswalk',
    'stairs',
    'curb',
    'terrain',
    'obstacle',
  ];
  static const hazardNames = [
    'stairs_ahead',
    'curb_ahead',
    'road_ahead',
    'crosswalk_ahead',
    'obstacle_ahead',
  ];

  final int imageWidth, imageHeight;
  final Map<String, double> ratios;

  /// Row 0 is far/top; row 2 is near/bottom.
  final List<List<WalkableCell>> grid;
  final bool pathClear, cached;
  final String suggest;
  final List<String> hazards;
  final int? maskWidth, maskHeight;
  final String? maskPngBase64;
  final Map<String, dynamic>? maskRle, diagnostics;

  factory WalkableResult.fromJson(Map<String, dynamic> json) {
    if (json['success'] != true ||
        json['path_clear'] is! bool ||
        json['cached'] is! bool ||
        !const [
          'straight',
          'left',
          'right',
          'stop',
        ].contains(json['suggest'])) {
      throw const FormatException('Invalid segmentation summary.');
    }
    final names = json['classes'];
    if (names is! List || names.length != classes.length) {
      throw const FormatException('Invalid segmentation classes.');
    }
    for (var i = 0; i < classes.length; i++) {
      if (names[i] != classes[i]) {
        throw const FormatException('Invalid segmentation class order.');
      }
    }
    final ratios = json['ratios'];
    final grid = json['grid'];
    final hazards = json['hazards'];
    if (ratios is! Map ||
        ratios.length != classes.length ||
        grid is! List ||
        grid.length != 3 ||
        hazards is! List ||
        hazards.any((h) => !hazardNames.contains(h))) {
      throw const FormatException('Invalid segmentation ratios/grid/hazards.');
    }
    final rows = <List<WalkableCell>>[];
    for (final row in grid) {
      if (row is! List || row.length != 3) {
        throw const FormatException('Expected a 3x3 grid.');
      }
      rows.add(
        List.unmodifiable(
          row.map((cell) {
            if (cell is! Map || !classes.contains(cell['dominant'])) {
              throw const FormatException('Invalid grid cell.');
            }
            return WalkableCell(
              walkable: _ratio(cell['walkable']),
              caution: _ratio(cell['caution']),
              blocked: _ratio(cell['blocked']),
              dominant: cell['dominant'] as String,
            );
          }),
        ),
      );
    }
    final hasMask = [
      'mask_w',
      'mask_h',
      'mask_png_b64',
      'mask_rle',
    ].any(json.containsKey);
    if (hasMask &&
        ((json['mask_png_b64'] is! String &&
                json['mask_rle'] is! Map<String, dynamic>) ||
            (json.containsKey('mask_png_b64') &&
                json.containsKey('mask_rle')))) {
      throw const FormatException('Invalid mask payload.');
    }
    final diagnostics = json['diagnostics'];
    if (json.containsKey('diagnostics')) {
      if (diagnostics is! Map<String, dynamic> ||
          diagnostics['source'] != 'colab' ||
          diagnostics['model'] is! String ||
          diagnostics['runtime'] is! String) {
        throw const FormatException('Invalid segmentation diagnostics.');
      }
      _nonnegative(diagnostics['inference_ms']);
      if (json['cached'] == true) _nonnegative(diagnostics['cache_age_ms']);
    }
    return WalkableResult._(
      imageWidth: _dimension(json['image_width']),
      imageHeight: _dimension(json['image_height']),
      ratios: Map.unmodifiable({
        for (final name in classes) name: _ratio(ratios[name]),
      }),
      grid: List.unmodifiable(rows),
      pathClear: json['path_clear'] as bool,
      suggest: json['suggest'] as String,
      hazards: List<String>.unmodifiable(hazards),
      cached: json['cached'] as bool,
      maskWidth: hasMask ? _dimension(json['mask_w']) : null,
      maskHeight: hasMask ? _dimension(json['mask_h']) : null,
      maskPngBase64: json['mask_png_b64'] as String?,
      maskRle: json['mask_rle'] as Map<String, dynamic>?,
      diagnostics: diagnostics as Map<String, dynamic>?,
    );
  }

  static int _dimension(dynamic value) {
    if (value is! int || value <= 0) {
      throw const FormatException('Expected a positive image dimension.');
    }
    return value;
  }

  static double _nonnegative(dynamic value) {
    if (value is! num || !value.isFinite || value < 0) {
      throw const FormatException('Expected a finite nonnegative number.');
    }
    return value.toDouble();
  }

  static double _ratio(dynamic value) {
    final number = _nonnegative(value);
    if (number > 1) throw const FormatException('Ratio outside [0,1].');
    return number;
  }
}

class WalkableCell {
  const WalkableCell({
    required this.walkable,
    required this.caution,
    required this.blocked,
    required this.dominant,
  });
  final double walkable, caution, blocked;
  final String dominant;
}
