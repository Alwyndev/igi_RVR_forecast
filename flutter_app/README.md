IGI RVR Flutter App

This is a minimal Flutter app that fetches multi-horizon RVR predictions from the local backend and displays them on a map with a horizon slider.

Quick start

1. Start the Python backend locally (from project root):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

2. Run the Flutter app (Android emulator recommended):

```bash
cd flutter_app
flutter pub get
flutter run
```

Notes
- For Android emulator, the app hits `http://10.0.2.2:5000/predictions_multi` by default. For a physical device, replace `backendUrl` in `lib/main.dart` with your machine IP `http://192.168.x.y:5000/predictions_multi`.
- The backend endpoint may take time to initialize because it loads models; consider adding caching or a lightweight precomputed JSON for production.
