import numpy as np
import cv2 as cv

import pytesseract as pyte
from PIL import ImageGrab


#after downloading pytesseract, add file to your path with the below code
pyte.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

text_rec = pyte.image_to_string("/path/to/file/takeoff.jpg")

print(text_rec)

