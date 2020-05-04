import os
import sys
import requests
# If you are using a Jupyter notebook, uncomment the following line.
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO

endpoint = "https://westcentralus.api.cognitive.microsoft.com/vision"

key_1 = "bd0ced714ad24db8b8c7fb110c18aa77"

key_2 = "e5668c29fa174b85a7c80a53e38d84f7"

api_key = key_1

ocr_url = endpoint + "vision/v2.1/ocr"

# Set image_url to the URL of an image that you want to analyze.
image_url = "images/1.jpg"
# Read the image into a byte array
image_data = open(image_url, "rb").read()

headers = {'Ocp-Apim-Subscription-Key': api_key}
# Set the langauge that you want to recognize. The value can be "en" for English, and "vi" for Vietnamese
# more info: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/language-support
params = {'language': 'vi', 'detectOrientation': 'true'}
response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
response.raise_for_status()

analysis = response.json()

# Extract the word bounding boxes and text.
line_infos = [region["lines"] for region in analysis["regions"]]
word_infos = []
for line in line_infos:
    for word_metadata in line:
        for word_info in word_metadata["words"]:
            word_infos.append(word_info)
word_infos
