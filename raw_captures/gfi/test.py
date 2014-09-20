#!/usr/bin/env python
import requests
import fileinput, json;

params = {'apikey': 'bcd84684331245c7a1b1a5f5f76a67d3c339f53c070a5a2b9129564de0655ec2', 'resource': '99017f6eebbac24f351415dd410d522d'}
response = requests.get('http://www.virustotal.com/vtapi/v2/file/report', params=params)
json_response = response.json()
print json_response


#print(json.dumps(json.loads("".join(json_response)),sort_keys=True, indent=4))

