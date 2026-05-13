import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_map_marker_cluster/flutter_map_marker_cluster.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;

// NOTE: Using host machine's local IP address (192.168.1.42) for physical device testing
// (Or 127.0.0.1 since we setup `adb reverse tcp:5000 tcp:5000`)
const String backendUrl = 'http://127.0.0.1:5000/predictions_multi';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  showdebugmodeBanner() => false;
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IGI RVR Map',
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
      appBar: AppBar(title: const Text('IGI RVR Map')),
      body: Column(
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: 100),
              child: Stack(
                children: [
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
                        userAgentPackageName: 'com.example.app',
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
                            final preds =
                                Map<String, dynamic>.from(z['predictions']);
                            final label = horizons.isNotEmpty
                                ? horizons[horizonIndex]
                                : '';
                            final value = preds[label]?.round() ?? 0;
                            final color = _colourForRvr(value);

                            return Marker(
                              point: LatLng(lat.toDouble(), lon.toDouble()),
                              width: 140,
                              height: 100,
                              builder: (context) => FittedBox(
                                fit: BoxFit.scaleDown,
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      width: 36,
                                      height: 36,
                                      decoration: BoxDecoration(
                                        color: color,
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                            color: Colors.white, width: 2),
                                        boxShadow: const [
                                          BoxShadow(
                                              blurRadius: 6,
                                              color: Colors.black45)
                                        ],
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    SizedBox(
                                      height: 26,
                                      child: Container(
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 6, vertical: 2),
                                        decoration: BoxDecoration(
                                            color: Colors.black54,
                                            borderRadius:
                                                BorderRadius.circular(4)),
                                        alignment: Alignment.center,
                                        child: Text(
                                          '${z['id']}\n${value}m',
                                          textAlign: TextAlign.center,
                                          maxLines: 2,
                                          overflow: TextOverflow.ellipsis,
                                          style: const TextStyle(fontSize: 10),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          }).toList(),
                          builder: (context, markers) {
                            return Container(
                              decoration: BoxDecoration(
                                color: Colors.blue.withOpacity(0.8),
                                shape: BoxShape.circle,
                                border:
                                    Border.all(color: Colors.white, width: 2),
                              ),
                              child: Center(
                                child: Text(
                                  markers.length.toString(),
                                  style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold),
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                  Positioned(
                    left: 8,
                    bottom: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                          color: Colors.black54,
                          borderRadius: BorderRadius.circular(4)),
                      child: const Text('IGI RVR',
                          style: TextStyle(color: Colors.white, fontSize: 12)),
                    ),
                  ),
                ],
              ),
            ),
          ),
          SafeArea(
            top: false,
            child: Container(
              color: Colors.black87,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        horizons.isNotEmpty ? horizons[horizonIndex] : '-',
                        style: const TextStyle(fontSize: 16),
                      ),
                      ElevatedButton.icon(
                        onPressed: fetchData,
                        icon: const Icon(Icons.refresh),
                        label: const Text('Refresh'),
                      ),
                    ],
                  ),
                  Slider(
                    value: horizonIndex.toDouble(),
                    min: 0,
                    max: (horizons.isNotEmpty ? (horizons.length - 1) : 0)
                        .toDouble(),
                    divisions: horizons.isNotEmpty ? (horizons.length - 1) : 1,
                    onChanged: (v) {
                      setState(() => horizonIndex = v.round());
                    },
                  ),
                ],
              ),
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
