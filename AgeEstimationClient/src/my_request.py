import json
import requests

host = "127.0.0.1"
port = "8023"


def detect(data):
    url = "http://%s:%s/detect" % (host, port)
    payload = json.dumps(data)
    headers = {'content-type': "application/json"}
    response = requests.request("POST", url, data=payload, headers=headers)
    return json.loads(response.text)
