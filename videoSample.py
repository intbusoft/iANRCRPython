import logging

import iANRCR
import iANRCRSettings

import cv2
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG)

config = iANRCRSettings.iANRCRConfig()
ia = iANRCR.iANRCR(config)

path = 'F:\\datasets\\vagons\\20200117_101554.mp4'
# divideW = 1 - обычное изображение, 2 - слишком вытянутое, по горизонтали разрезать надвое
divideW = 1

cap = cv2.VideoCapture(path)

out = None

def draw_number(numbers,numbers_memory,img):
    if numbers is None or numbers == []:
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX     
    w = img.shape[1]

    for n in numbers:
        if n is None or n == []:
            continue

        x1 = n[2][1]
        y1 = n[2][0]
        x2 = n[2][3]
        y2 = n[2][2]
        
        img = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255),2)
        s = '{}|{:.2f}'.format(n[0],n[1])
                    
        yt = int(y1)-10
        if yt < 30:
            yt = int(y2)+30
        textSize = cv2.getTextSize(s, fontFace=font, fontScale=1, thickness=2)
        if x1 + textSize[0][0] > w:
            x1 = w - textSize[0][0]
        img = cv2.putText(img,s,(int(x1),yt),font,1,(0,0,255),2)

    if numbers_memory != None and numbers_memory != []:
        for i,n in enumerate(numbers_memory):
            s = '{}'.format(n)
            textSize = cv2.getTextSize(s, fontFace=font, fontScale=1, thickness=2)
            img = cv2.rectangle(img, (0,textSize[0][1]*i), (textSize[0][0],textSize[0][1]*(i+1)), (255,255,255),cv2.FILLED)
            img = cv2.putText(img,s,(0,textSize[0][1]*(i+1)),font,1,(0,0,0),2)
    return img

err = 0
while cap.isOpened():
     ret, frame = cap.read()         
     if ret:
         if out is None:
             out = cv2.VideoWriter('outvideo.avi',cv2.VideoWriter_fourcc('D','I','V','X'), 30, (frame.shape[1]//divideW,frame.shape[0]))
         if divideW == 1:
             image = frame.copy()
         if divideW == 2:
             image = frame[:,0:frame.shape[1]//divideW]
         t1 = time.time()
         ia.process([image])
         t2 = time.time()
         print("{:.3f}".format(t2-t1))
         #images = ia.draw_symbols([frame],None)
         mem = ia.get_numbers_memory()
         if mem != None:
             mem = mem[0]
         image = draw_number(ia.get_numbers()[0],mem,image)
         cv2.imshow("Test iANRCR",image)
         out.write(image)
         c = cv2.waitKey(1)
         if c & 0xFF == ord('q'):
             break
         err = 0
     err += 1
     if err > 1000:
        break


cap.release()
out.release()

cv2.destroyAllWindows()