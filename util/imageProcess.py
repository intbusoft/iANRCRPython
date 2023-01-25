'''
 Модуль для обработки изображений и работы с Rectами и Boxами
'''
import numpy as np
import cv2
from dataclasses import dataclass

def check_nestedBox(box1,box2):
    '''
     Проверить вложен ли box2 в box1. 
     Возвращаеn bool
    '''
    return box2[0] >= box1[0] and box2[1] >= box1[1] and box2[2] <= box1[2] and box2[3] <= box1[3]

def check_intersectionP(x1,x2):
    '''
     Проверить пересечение , где x1,x2 - пары [левое правое]
     Возвращает пересечение
    '''
    return min(x1[1],x2[1]) - max(x1[0],x2[0])

def check_intersectionBox(box1,box2):
    '''
     Проверить пересечение коробок
     box - [y1,x1,y2,x2,...]
     Возвращает box пересечения и статус
    '''
    x1 = max(box1[1],box2[1])
    y1 = max(box1[0],box2[0])
    x2 = min(box1[3],box2[3])
    y2 = min(box1[2],box2[2])
    if y2 < y1 or x2 < x1:
        return False, None
    return True,[y1,x1,y2,x2]
