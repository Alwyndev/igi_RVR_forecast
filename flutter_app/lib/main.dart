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

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IGI RVR Map',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const MapPage(),
    );
  }
}

class MapPage extends StatefulWidget {
  const MapPage({super.key});
  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  Map<String, dynamic>? data;
  int horizonIndex = 0;
  List<String> horizons = [];

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
            options: MapOptions(
              center: LatLng(28.555, 77.095),
              zoom: 14.0,
            ),
            children: [
              TileLayer(
                urlTemplate:
                    'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
                subdomains: const ['a', 'b', 'c', 'd'],
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
                // 50% opacity black background
                color: Colors.black.withOpacity(0.5),
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
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                            letterSpacing: 1.1,
                          ),
                        ),
                        ElevatedButton.icon(
                          onPressed: fetchData,
                          icon: const Icon(Icons.refresh),
                          label: const Text('Refresh'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.white24,
                            foregroundColor: Colors.white,
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
                      inactiveColor: Colors.white30,
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
          Positioned(
            left: 16,
            top: 48, // offset for device status bar
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.white24)),
              child: const Text('IGI RVR Forecaster',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.bold)),
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
