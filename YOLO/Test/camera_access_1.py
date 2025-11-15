import numpy as np
import cv2 as cv
import pytesseract as pytess
from PIL import ImageGrab


#after downloading pytesseract, add file to your path with the below code
pytess.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


# Run the code below to see accessible device list
# ffmpeg -f avfoundation -list_devices true -i ""


cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 20)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    text_rec = pytess.image_to_string(frame)
    
    detected = pytess.image_to_data(frame, output_type = pytess.Output.DICT)
    n_boxes = len(detected['text'])
    for i in range(n_boxes):
        if int(detected['conf'][i]) > 0:
            (x,y,w,h) = (detected['left'][i], detected['top'][i], detected['width'][i], detected['height'][i])
            frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0,255, 0), 2)
        
    frame_with_text = frame.copy()
    frame_with_text = cv.putText(frame_with_text, text_rec, (10,30), cv.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2) 

    cv.imshow('Frame found with Text', frame_with_text)

    if cv.waitKey(1) & 0xFF == ord('q'):
                break


cap.release()
cv.destroyAllWindows()


