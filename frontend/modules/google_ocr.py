import cv2
import json
import numpy as np
import pandas as pd
import requests
from base64 import b64encode

def makeImageData(imgpath):
    img_req = None
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()

def requestOCR(url, api_key, imgpath):
  imgdata = makeImageData(imgpath)
  response = requests.post(ENDPOINT_URL,
                           data = imgdata,
                           params = {'key': api_key},
                           headers = {'Content-Type': 'application/json'})
  return response

with open('modules/vision_api.json') as f:
    data = json.load(f)

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
api_key = data["api_key"]

def read_image(img_loc):

    RES = ''

    result = requestOCR(ENDPOINT_URL, api_key, img_loc)

    if result.status_code != 200 or result.json().get('error'):
        print(result.text)
        print("Error")
    else:
        result = result.json()['responses'][0]['textAnnotations']

    for index in range(len(result)):
      RES += (result[index]["description"])

    return RES