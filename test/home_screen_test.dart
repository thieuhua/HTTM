import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:smart_navigation/controllers/session_controller.dart';
import 'package:smart_navigation/models/route_progress.dart';
import 'package:smart_navigation/screens/home_screen.dart';
import 'session_navigation_test.dart' show FakeRoute, RecordingTts, NoCamera;

class PreviewSession extends SessionController {
  PreviewSession()
    : super(
        routeService: FakeRoute(),
        ttsService: RecordingTts(),
        frameSource: NoCamera(),
      );
  @override
  Future<void> initialize() async {}
  @override
  Future<bool> start() async => false;
  @override
  RouteProgress get routeSnapshot => const RouteProgress(
    status: NavigationStatus.idle,
    message:
        'Đã tìm thấy vị trí. Chọn điểm đến hoặc nói đi đến địa điểm muốn tới.',
  );
}

void main() {
  testWidgets(
    'main screen supports a keyboard and enlarged text without overflow',
    (tester) async {
      tester.view.physicalSize = const Size(360, 640);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final session = PreviewSession();
      await tester.pumpWidget(
        MaterialApp(
          builder: (context, child) => MediaQuery(
            data: MediaQuery.of(context).copyWith(
              textScaler: const TextScaler.linear(1.5),
              viewInsets: const EdgeInsets.only(bottom: 280),
            ),
            child: child!,
          ),
          home: HomeScreen(controller: session),
        ),
      );
      await tester.pump();
      expect(find.byType(TextField), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox());
      await tester.pump();
    },
  );
}
