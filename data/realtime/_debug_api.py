import requests
import re
import json

url = 'http://rvrcamd.imd.gov.in:5000/assets/index-DNaPqE4r.js'
try:
    r = requests.get(url, timeout=10)
    js = r.text
    
    # Let's search for endpoint keywords
    endpoints = re.findall(r'\"(/[a-zA-Z0-9_-]+/[^\"]+)\"', js)
    endpoints = list(set(endpoints))
    
    print("Found potential endpoints:")
    for ep in endpoints:
        if 'api' in ep or 'rvr' in ep.lower():
            print(ep)
            
    # Also let's try calling a few endpoints without auth
    base_url = "http://103.215.208.153:8444"
    test_eps = ["/api/public/rvr", "/public/rvr", "/api/rvr/public", "/live-rvr"]
    for ep in test_eps:
        url = f'{base_url}{ep}'
        try:
            res = requests.get(url, timeout=5)
            print(f'[{res.status_code}] {url}')
            if res.status_code == 200:
                print(res.text[:200])
        except Exception as e:
            pass
except Exception as e:
    print(e)
