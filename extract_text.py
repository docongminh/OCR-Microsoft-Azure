import os
import sys
import requests
# If you are using a Jupyter notebook, uncomment the following line.
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO

def check_exists(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        print(e)
        pass


endpoint = "https://westcentralus.api.cognitive.microsoft.com/vision"

key_1 = "bd0ced714ad24db8b8c7fb110c18aa77"

key_2 = "e5668c29fa174b85a7c80a53e38d84f7"

api_key = key_1

ocr_url = endpoint + "vision/v2.1/ocr"

# Set image_url to the URL of an image that you want to analyze.
list_image_url = "images/1.jpg"
path_save_ann = "extract_text/annotation"
path_save_words = "extract_text/words_bboxs"
check_exists(path_save_ann)
check_exists(path_save_words)

for image_url in list_image_url:

    # Read the image into a byte array
    image_data = open(image_url, "rb").read()

    headers = {'Ocp-Apim-Subscription-Key': api_key}
    # Set the langauge that you want to recognize. The value can be "en" for English or "vi" for Vietnamese
    # more info: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/language-support
    params = {'language': 'vi', 'detectOrientation': 'true'}
    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()

    analysis = response.json()
    #
    image_name = image_url.split("/")[-1]
    _n = os.path.splitext(image_name)[0]
    # image to write result
    img = cv2.imread(image_url)
    word_image = img.copy()
    # extract information
    lines = analysis["regions"]["lines"]

    with open("{}/{}.txt".format(path_save_ann, _n), 'w') as f:
        for line in lines:
            text = line["words"]["text"]
            bbox = line["words"]["boundingBox"]
            bboxs = bbox.split(",")
            x, y, xx, yy = int(bboxs[0]), int(bboxs[1]), int(bboxs[2]), int(bboxs[3])
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
