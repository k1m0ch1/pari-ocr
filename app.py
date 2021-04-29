from fastapi import FastAPI, Request, File, UploadFile
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from imutils.perspective import four_point_transform
from imutils import contours
from datetime import datetime
import uvicorn
import os
import imutils
import numpy as np
import io
import cv2
import pytesseract
import sys


class ImageType(BaseModel):
 url: str

if not os.path.isdir("./statics"):
    os.mkdir("./statics")
    os.mkdir("./statics/uploads")

if not os.path.isdir("./statics/uploads"):
    os.mkdir("./statics/uploads")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/statics", StaticFiles(directory="statics"), name="statics")

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

ADIGITS_LOOKUP = {
	(0, 0, 0, 0, 0, 0, 0): 0,
	(1, 1, 1, 0, 1, 0, 1): 0,
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 0, 0, 1, 0, 0): 1,
	(1, 0, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 1, 0, 0, 1): 3,
    (1, 1, 0, 0, 1, 1, 1): 3,
	(1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 0, 0, 1, 1): 5,
	(1, 0, 1, 0, 1, 1, 1): 2,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9,
    (1, 1, 1, 0, 0, 1, 1): 9,
}

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def read_img(img, commands):
    now = datetime.now()
    image = img
    cv2.imwrite('./statics/original.jpg', image)
    if commands == "store":
        cv2.imwrite(f'./statics/uploads/{datetime.timestamp(now)}-original.jpg', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    # cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	# cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # displayCnt = None

    # # loop over the contours
    # for c in cnts:
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     ea = cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,0),2)
    #     cv2.imwrite('./lol.jpg', ea)
    #     # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the contour has four vertices, then we have found
        # the thermostat display
        # if len(approx) == 4:
        #     displayCnt = approx
        #     break

    # warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    # output = four_point_transform(image, displayCnt.reshape(4, 2))
    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    thresh = cv2.threshold(edged, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cv2.imwrite('./statics/threshold.jpg', thresh)
    if commands == "store":
        cv2.imwrite(f'./statics/uploads/{datetime.timestamp(now)}-threshold.jpg', thresh)

    # thresh = cv2.bitwise_not(thresh)

    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    print(len(cnts))
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # lol = input(f"{x} {y} {w} {h} enter to continue")
        print(f"{x} {y} {w} {h} enter to continue")
        # if the contour is sufficiently large, it must be a digit
        if w >= 71 and h >= 19:
            ea = cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.imwrite('./statics/compare.jpg', ea)
            if commands == "store":
                cv2.imwrite(f'./statics/uploads/{datetime.timestamp(now)}-compare.jpg', ea)
            digitCnts.append(c)

    if commands == "comparing":
            return

    if len(digitCnts) == 0:
        print("I got Nothing")
        return 
    # sort the contours from left-to-right, then initialize the
    # actual digits themselves
    digitCnts = contours.sort_contours(digitCnts,
        method="left-to-right")[0]
    digits = []

    print(len(digitCnts))
    hasil = ""
    # loop over each of the digits
    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = edged[y:y + h, x:x + w]
        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)
        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),	# top
            ((0, 0), (dW, h // 2)),	# top-left
            ((w - dW, 0), (w, h // 2)),	# top-right
            ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
            ((0, h // 2), (dW, h)),	# bottom-left
            ((w - dW, h // 2), (w, h)),	# bottom-right
            ((0, h - dH), (w, h))	# bottom
        ]

        on = [0] * len(segments)
    
        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            print(f"total {total}")
            area = (xB - xA) * (yB - yA)
            print(f"area {area}")
            print(f"ttt {total / float(area)}")
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if total / float(area) > 0.05:
                on[i]= 1

        # import pdb; pdb.set_trace()

        if tuple(on) in ADIGITS_LOOKUP:
            # lookup the digit and draw it on the image
            print(f"{on} {ADIGITS_LOOKUP[tuple(on)]}")
            digit = ADIGITS_LOOKUP[tuple(on)]
            digits.append(digit)
            hasil += str(digit)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(image, str(digit), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imwrite('./statics/result.jpg', image)
            if commands == "store":
                cv2.imwrite(f'./statics/uploads/{datetime.timestamp(now)}-result.jpg', image)
        else:
            return "NOT FOUND"
    
    return hasil
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Otsu Tresholding automatically find best threshold value
    # _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # # invert the image if the text is white and background is black
    # count_white = np.sum(binary_image > 0)
    # count_black = np.sum(binary_image == 0)
    # if count_black > count_white:
    #     binary_image = 255 - binary_image
        
    # # padding
    # final_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # txt = pytesseract.image_to_string(
    #     final_image, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
    # return(txt)

#  , file: bytes = File(...)

@app.post("/extract_text") 
async def extract_text(request: Request):
    label = ""
    if request.method == "POST":
        form = await request.form()
        # file = form["upload_file"].file
        contents = await form["upload_file"].read()
        image_stream = io.BytesIO(contents)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label =  read_img(frame, form["commands"])
       
        # return {"label": label}

    if form["commands"] == "store":
        return {"message": "success"}

    if form["commands"] == "comparing":
        return templates.TemplateResponse("compare.html", {"request": request})
    return templates.TemplateResponse("index.html", {"request": request, "label": label})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="info")