import requests
import re
import json

url = 'http://rvrcamd.imd.gov.in:5000/assets/index-DNaPqE4r.js'
try:
    r = requests.get(url, timeout=10)
    js = r.text
    
    idx = js.find('live-rvr')
    if idx > -1:
        print("Found 'live-rvr' in JS:")
        print(js[max(0, idx-100) : min(len(js), idx+500)])
        
    print("\n--- Testing specific endpoints ---")
    base_url = "http://103.215.208.153:8444"
    eps = [
        "/wc/internal/api/rvr/data",
        "/wc/internal/api/weather/display/data",
        "/api/weather/display/data"
    ]
    for ep in eps:
        url = f'{base_url}{ep}'
        print(f"GET {url}")
        res = requests.get(url, timeout=5)
        print(f"[{res.status_code}] {res.text[:100]}")
except Exception as e:
    print(e)
