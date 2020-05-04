import json
import os
import sys
import requests
import time
# If you are using a Jupyter notebook, uncomment the following line.
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from io import BytesIO


def check_exists(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        raise e


endpoint = "https://westcentralus.api.cognitive.microsoft.com/vision"

key_1 = "bd0ced714ad24db8b8c7fb110c18aa77"

key_2 = "e5668c29fa174b85a7c80a53e38d84f7"

api_key = key_1


text_recognition_url = endpoint + "/vision/v3.0-preview/read/analyze"

# Set image_url to the URL of an image that you want to recognize.
list_image_url = "image path"
path_save_ann = "extract_hand_text/annotation"
path_save_words = "extract_hand_text/words_bboxs"
check_exists(path_save_ann)
check_exists(path_save_words)

for image_url in list_image_url:
    # Set the langauge that you want to recognize. The value can be "en" for English, and "vi" for Vietnamese
    # more info: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/language-support
    language = "vi"
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    # Read the image into a byte array
    image_data = open(image_url, "rb").read()
    response = requests.post(
        text_recognition_url, headers=headers, data=image_data, params={'language': language})
    response.raise_for_status()

    # Extracting text requires two API calls: One call to submit the
    # image for processing, the other to retrieve the text found in the image.

    # Holds the URI used to retrieve the recognized text.
    operation_url = response.headers["Operation-Location"]

    # The recognized text isn't immediately available, so poll to wait for completion.
    analysis = {}
    poll = True
    while (poll):
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        
        print(json.dumps(analysis, indent=4))

        time.sleep(1)
        if ("analyzeResult" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'failed'):
            poll = False

    #
    image_name = image_url.split("/")[-1]
    _n = os.path.splitext(image_name)[0]
    # image to write result
    img = cv2.imread(image_url)
    word_image = img.copy()

    if ("analyzeResult" in analysis):
        # Extract the recognized text, with bounding boxes.
        lines = analysis["analyzeResult"]["readResults"][0]
        with open("{}/{}.txt".format(path_save_ann, _n), 'w') as f:
            for line in lines:
                for word in line["words"]:
                    text = word["text"]
                    bbox = word["boundingBox"]
                    x, y, xx, yy = int(bbox[0]), int(bbox[1]), int(bbox[4]), int(bbox[5])
                    word_rect = cv2.rectangle(word_image, (x, y), (xx, yy), (255, 0, 255), 2)
                    f.write(text)
                    f.write("\t")
                    f.write(str(x))
                    f.write("\t")
                    f.write(str(y))
                    f.write("\t")
                    f.write(str(xx))
                    f.write("\t")
                    f.write(str(yy))
                    f.write("\n")
        cv2.imwrite('{}/word_rect_{}.jpg'.format(path_save_words, _n), word_rect)



        

