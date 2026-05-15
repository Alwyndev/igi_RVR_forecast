import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_map_marker_cluster/flutter_map_marker_cluster.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';

// Production Cloud Run endpoint
const String backendUrl =
    'https://igi-rvr-api-969804968558.asia-south1.run.app/predictions_multi';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  ThemeMode _themeMode = ThemeMode.light;

  void toggleTheme() {
    setState(() {
      _themeMode =
          _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IGI RVR',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.light(),
      darkTheme: ThemeData.dark(),
      themeMode: _themeMode,
      home: MapPage(onToggleTheme: toggleTheme, currentTheme: _themeMode),
    );
  }
}

class MapPage extends StatefulWidget {
  final VoidCallback onToggleTheme;
  final ThemeMode currentTheme;

  const MapPage(
      {super.key, required this.onToggleTheme, required this.currentTheme});

  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  Map<String, dynamic>? data;
  int horizonIndex = 0;
  List<String> horizons = [];
  String generatedAt = '';
  final MapController _mapController = MapController();
  bool _showRecenter = false;

  @override
  void initState() {
    super.initState();
    fetchData();
  }

  Future<void> fetchData() async {
    try {
      final resp = await http.get(Uri.parse(backendUrl));
      if (resp.statusCode == 200) {
        final payload = jsonDecode(resp.body) as Map<String, dynamic>;
        setState(() {
          data = payload;
          horizons = List<String>.from(payload['horizons'] ?? []);
          generatedAt = payload['generated_at'] ?? '';
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Backend error: ${resp.statusCode}')));
        debugPrint('Backend error: ${resp.statusCode}');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Fetch failed: $e')));
      }
      debugPrint('Fetch failed: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final zones =
        data != null ? List<Map<String, dynamic>>.from(data!['zones']) : [];

    return Scaffold(
      extendBodyBehindAppBar: true,
      body: Stack(
        children: [
          // Background Map Layer covers the entire screen
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              center: LatLng(28.555, 77.095),
              zoom: 14.0,
              minZoom: 12.5,
              maxZoom: 18.0,
              onPositionChanged: (pos, hasGesture) {
                if (pos.center != null) {
                  final dist = const Distance().as(
                      LengthUnit.Kilometer, pos.center!, LatLng(28.555, 77.095));
                  if ((dist > 3.0) != _showRecenter) {
                    setState(() {
                      _showRecenter = dist > 3.0;
                    });
                  }
                }
              },
            ),
            children: [
              TileLayer(
                urlTemplate: widget.currentTheme == ThemeMode.dark
                    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'
                    : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                subdomains: const ['a', 'b', 'c'],
                userAgentPackageName: 'com.igi.antigravity',
                errorTileCallback: (tile, error, stackTrace) {
                  // Suppresses unhandled error logs for occasionally dropped tile connections
                },
              ),
              MarkerClusterLayerWidget(
                options: MarkerClusterLayerOptions(
                  maxClusterRadius: 45,
                  size: const Size(40, 40),
                  fitBoundsOptions: const FitBoundsOptions(
                    padding: EdgeInsets.all(50),
                  ),
                  markers: zones.map((z) {
                    final lat = z['lat'] as num;
                    final lon = z['lon'] as num;
                    final preds = Map<String, dynamic>.from(z['predictions']);
                    final label =
                        horizons.isNotEmpty ? horizons[horizonIndex] : '';
                    final value = preds[label]?.round() ?? 0;
                    final color = _colourForRvr(value);

                    return Marker(
                      point: LatLng(lat.toDouble(), lon.toDouble()),
                      width: 140,
                      height: 80,
                      rotate: true,
                      builder: (context) => FittedBox(
                        fit: BoxFit.scaleDown,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 8),
                          decoration: BoxDecoration(
                            color: color,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: Colors.white, width: 2),
                            boxShadow: const [
                              BoxShadow(blurRadius: 6, color: Colors.black45)
                            ],
                          ),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                '${z['id']}',
                                style: const TextStyle(
                                  fontSize: 14,
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                '${value}m',
                                style: const TextStyle(
                                  fontSize: 18,
                                  color: Colors.white,
                                  fontWeight: FontWeight.w900,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  }).toList(),
                  builder: (context, markers) {
                    return Container(
                      decoration: BoxDecoration(
                        color: Colors.blue.withOpacity(0.8),
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 2),
                      ),
                      child: Center(
                        child: Text(
                          markers.length.toString(),
                          style: const TextStyle(
                              color: Colors.white, fontWeight: FontWeight.bold),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),

          // Transparent Overlay Slider Control at the Bottom
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: SafeArea(
              top: false,
              child: Container(
                // Transparent background
                color: Colors.transparent,
                padding:
                    const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          horizons.isNotEmpty
                              ? 'Forecast: ${horizons[horizonIndex]}'
                              : 'Loading...',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: widget.currentTheme == ThemeMode.dark
                                ? Colors.white
                                : Colors.black87,
                            letterSpacing: 1.1,
                          ),
                        ),
                        IconButton(
                          onPressed: fetchData,
                          icon: const Icon(Icons.refresh),
                          tooltip: 'Refresh Data',
                          color: widget.currentTheme == ThemeMode.dark
                              ? Colors.white
                              : Colors.black87,
                          style: IconButton.styleFrom(
                            backgroundColor: widget.currentTheme == ThemeMode.dark
                                ? Colors.white10
                                : Colors.black12,
                          ),
                        ),
                      ],
                    ),
                    Slider(
                      value: horizonIndex.toDouble(),
                      min: 0,
                      max: (horizons.isNotEmpty ? (horizons.length - 1) : 0)
                          .toDouble(),
                      divisions:
                          horizons.isNotEmpty ? (horizons.length - 1) : 1,
                      activeColor: Colors.blueAccent,
                      inactiveColor: Colors.black45,
                      onChanged: (v) {
                        setState(() => horizonIndex = v.round());
                      },
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Custom Floating App Title at Top Left since AppBar is removed
          // Custom Floating App Title at Top Left since AppBar is removed
          Positioned(
            left: 16,
            top: 48, // offset for device status bar
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                      color: widget.currentTheme == ThemeMode.dark
                          ? Colors.black87
                          : Colors.white,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                          color: widget.currentTheme == ThemeMode.dark
                              ? Colors.white24
                              : Colors.black12)),
                  child: Row(
                    children: [
                      Text('IGI RVR',
                          style: TextStyle(
                              color: widget.currentTheme == ThemeMode.dark
                                  ? Colors.white
                                  : Colors.black87,
                              fontSize: 16,
                              fontWeight: FontWeight.bold)),
                      const SizedBox(width: 16),
                      IconButton(
                        icon: Icon(
                          widget.currentTheme == ThemeMode.dark
                              ? Icons.light_mode
                              : Icons.dark_mode,
                          color: widget.currentTheme == ThemeMode.dark
                              ? Colors.white
                              : Colors.black87,
                        ),
                        onPressed: widget.onToggleTheme,
                        tooltip: 'Toggle Theme',
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                if (generatedAt.isNotEmpty)
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                    decoration: BoxDecoration(
                      color: widget.currentTheme == ThemeMode.dark
                          ? Colors.black54
                          : Colors.white70,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      'Last Updated: ${(() {
                        String timeStr = generatedAt;
                        if (!timeStr.endsWith('Z') && !timeStr.contains('+')) {
                          timeStr += 'Z';
                        }
                        DateTime dt = DateTime.parse(timeStr).toUtc();
                        return dt.add(const Duration(hours: 5, minutes: 30)).toString().split('.')[0];
                      })()} IST',
                      style: TextStyle(
                        color: widget.currentTheme == ThemeMode.dark
                            ? Colors.white70
                            : Colors.black87,
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
              ],
            ),
          ),

          // Recenter Button - appears when user pans far away
          if (_showRecenter)
            Positioned(
              right: 16,
              bottom: 120, // Positioned above the slider panel
              child: FloatingActionButton(
                mini: true,
                backgroundColor: widget.currentTheme == ThemeMode.dark ? Colors.blueAccent : Colors.blue,
                child: const Icon(Icons.center_focus_strong, color: Colors.white),
                onPressed: () {
                  _mapController.move(LatLng(28.555, 77.095), 14.0);
                },
              ),
            ),
        ],
      ),
    );
  }

  Color _colourForRvr(int rvr) {
    if (rvr >= 1500) return Colors.green.shade800;
    if (rvr >= 550) return Colors.orange;
    if (rvr >= 175) return Colors.red;
    return Colors.black;
  }
}
