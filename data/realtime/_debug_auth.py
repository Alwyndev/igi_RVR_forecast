import requests
import re

url = 'http://rvrcamd.imd.gov.in:5000/assets/index-DNaPqE4r.js'
js = requests.get(url, timeout=10).text

public_eps = re.findall(r'\"(/[^\"]*public[^\"]*)\"', js)
print("Public Endpoints:", list(set(public_eps)))

token_eps = re.findall(r'\"(/[^\"]*token[^\"]*)\"', js)
print("Token Endpoints:", list(set(token_eps)))
