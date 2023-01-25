import logging

import iANRCR
import iANRCRSettings

import cv2
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG)

config = iANRCRSettings.iANRCRConfig()
ia = iANRCR.iANRCR(config)

def Test1Image():
    image = cv2.imread("imagetest1.jpg")
    ia.process([image])
    ia.draw_symbols([image],["imagetest1_out.jpg"])

def TestSomeImages():
    image1 = cv2.imread("imagetest1.jpg")
    image2 = cv2.imread("imagetest2.jpg")    
    ia.process([image1,image2])
    ia.draw_symbols([image1,image2],["imagetest1_out.jpg","imagetest2_out.jpg"])


def SpeedTest():
    # Тестирование быстродействия
    frame = cv2.imread("imagetest1.jpg")
    batch_sizes = [1,2,4,8]    
    test_it = 40
    end_it = 20
    res = []
    for batch in batch_sizes:
        mean = 0
        for i in range(test_it):
            frames=[frame for j in range(batch)]   
            start_time = time.time()
            ia.process(frames)
            end_time = time.time()
            print(end_time-start_time)
            if i >= test_it - end_it:
                mean = mean + (end_time-start_time)
        print("Mean:",mean/end_it,(mean/end_it)/batch)
        res.append((mean/end_it)/batch)
    print("Results:")
    [print("Batch size = ",batch,": ",res[i]) for i,batch in enumerate(batch_sizes)]

#Test1Image()
TestSomeImages()
#SpeedTest()