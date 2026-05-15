# IGI RVR Flutter App

A Flutter-based mobile map client for the IGIA RVR Forecasting system. Displays real-time multi-horizon visibility predictions on an interactive dark-mode map with ICAO category color-coding.

## Architecture

The app fetches predictions from the Cloud Run backend:

```
Flutter App  →  GET /predictions_multi  →  Cloud Run (Flask API)  →  V3/V5 Dynamic Hybrid Engine
```

## Prerequisites

- Flutter SDK (≥2.17.0)
- Android SDK (for Android builds)
- Xcode (for iOS builds)

## Quick Start

```bash
cd flutter_app
flutter pub get
flutter run
```

## Building a Release APK

```bash
flutter build apk --release
```

The output APK is at `build/app/outputs/flutter-apk/app-release.apk`.

## Configuration

### Backend URL

The production endpoint is configured in `lib/main.dart`:

```dart
const String backendUrl =
    'https://igi-rvr-api-969804968558.asia-south1.run.app/predictions_multi';
```

**For local development**, replace with:

- Android emulator: `http://10.0.2.2:5000/predictions_multi`
- Physical device (same Wi-Fi): `http://<your-machine-ip>:5000/predictions_multi`

### App Icon

The launcher icon is generated from `assets/RVR_logo.png` using `flutter_launcher_icons`:

```bash
flutter pub run flutter_launcher_icons
```

## Features

- **Edge-to-edge dark map** (CartoDB Dark Matter tiles)
- **Pill-shaped markers** with zone ID + RVR value, color-coded by ICAO fog category
- **Bottom overlay slider** to scrub through forecast horizons (10m → 6h)
- **Landscape-locked orientation** for optimal airfield viewing
- **Rotation-stable markers** (`rotate: true`) that stay upright during map interaction
