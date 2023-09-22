# iANRCR - основной файл для SDK распознавания номеров жд вагонов
import cv2
import tensorflow as tf
import numpy as np
import logging

import iANRCRSettings
import iANRCRDetect
import util.imageProcess as ipr

__version__ = '1.0.0'

class iANRCR(object):
    def __init__(self,config):   
        self.config = config
        self.Detection = iANRCRDetect.iANRCRDetect(config)
        self.list_objects = None
        self.numbersImage = None
        self.memoryNumbers = None

    def convert_detect_image(self,img,batch = 1):
        '''
         Конвертировать в изображение
        '''
        frame = cv2.resize(img, (self.config.detect_width, self.config.detect_height))
        frame = frame.astype(np.float32)
        frame /= 255.
        if batch == 1:
            frame = np.expand_dims(frame, 0) 
        return frame

    def process(self,images):
        '''
         Запуск процесса распознавания номеров на группе изображений
         images - группа изображений для распознавания в формате cv (uint), batch_size - размер этой группы
        '''         
        if images is None or images[0] is None:
            logging.warning('images == None')
            return                   

        im = None	
        if len(images) == 1:
            im = self.convert_detect_image(images[0])
        else:
            im = np.stack([self.convert_detect_image(i,len(images)) for i in images])              

        self.list_objects = self.Detection.process(im)
        
        if not self.list_objects  is None:
            # нормализация результатов и формирование номера
            self.numbersImage = []
            for j,img in enumerate(images):
                symbolsImage = []
                for symb in range(self.config.types_of_object_detection):                
                    if self.list_objects[symb][0][j] is None:
                        continue
                    h,w,c = img.shape
                    # Нормализация для изображения
                    dh = h/self.config.detect_height
                    dw = w/self.config.detect_width
                    self.list_objects[symb][0][j][..., :4] *= [dh, dw, dh, dw]
                    # Для каждого изображения формируем массив символов и передаем его на распознавания номера в кадре
                    for i,s in enumerate(self.list_objects[symb][0][j]):
                        symbolsImage.append([str(symb),s,self.list_objects[symb][1][j][i]])

                number = self.calc_numbers(symbolsImage)
                self.numbersImage.append(number)

            if self.numbersImage != []:
                self.add_in_memory()
        pass

    def add_in_memory(self):
        '''
         Записываются номера в очередь
        ''' 
        k = 0
        if self.memoryNumbers == None or len(self.numbersImage) != len(self.memoryNumbers):
            k = 1

        if self.memoryNumbers == None:
            self.memoryNumbers = []

        for i,n in enumerate(self.numbersImage):
            if k == 1:
                self.memoryNumbers.append([])
            self.memoryNumbers[i].append(n)
            if len(self.memoryNumbers[i]) > self.config.memory_number_frames:
                del self.memoryNumbers[i][0]

    def calc_numbers(self,symbolsImage):
        '''
         Из символов объединить в номера
        '''
        if symbolsImage == []:
            return None 

        symbolsImage = sorted(symbolsImage, key=lambda cand:cand[1][1])

        numbersSequence = []
        for i,s in enumerate(symbolsImage):
            if numbersSequence == []:
                numbersSequence.append([])
                numbersSequence[0].append(s)
            else:
                j = -1
                for k,n in enumerate(numbersSequence):
                    h = (s[1][2]-s[1][0])
                    if s[1][1] - n[-1][1][3] < self.config.max_distance_between_charactersW*h and \
                                abs(s[1][0] - n[-1][1][0]) < self.config.max_distance_between_charactersH*h and \
                                abs(h- (n[-1][1][2]-n[-1][1][0])) < h*.3:
                        j = k
                        break

                if j == -1:
                    # Попробовать добавить еще номер
                    numbersSequence.append([])
                    numbersSequence[-1].append(s)
                else:
                    # Проверить пересечение с предыдущим символом
                    s0 = numbersSequence[j][-1]
                    intersection = ipr.check_intersectionP([s0[1][1],s0[1][3]],[s[1][1],s[1][3]])
                    if intersection > (s0[1][3]-s0[1][1])/2 or intersection > (s[1][3]-s[1][1])/2:
                        # пересекаются - один нужно удалить
                        if s[2] > s0[2]:
                            numbersSequence[j][-1] = s
                    else:
                        numbersSequence[j].append(s)
        # Сформировать номера и достоверность
        numbersResult = []
        if numbersSequence != []:
            for n in numbersSequence:
                y1,y2 = n[0][1][0],n[0][1][2]
                x1,x2 = n[0][1][1],n[-1][1][3]
                s = ""
                score = 0.0
                for num in n:
                    s += num[0]
                    score += num[2]
                    if num[1][0] < y1:
                        y1 = num[1][0]
                    if num[1][2] > y2:
                        y2 = num[1][2]
                score /= len(s)

                if self.config.correct_number:
                    if not self.control_number(s):
                        continue

                numbersResult.append([s,score,(y1,x1,y2,x2)])

        return numbersResult

    def control_number(selg, num):
        '''
         Проверяет правильность номера по стандарту РЖД.
         Контрольная цифра(восьмая) – это цифра, дополняющая под
         разрядную сумму до ближайшего целого десятка. Программа
         проверяет таким образом правильность формирования номера.
        '''
        if len(num) != 8:
            return False

        sum = 0
        for i in range(7):
            a = int(num[i])* (2 - i%2)
            sum += (a//10) + (a%10)

        if sum % 10 == 0 and int(num[7]) != 0:
            return False

        if sum % 10 > 0 and int(num[7]) != 10 - (sum % 10):
            return False

        return True

    def get_numbers(self):
        return self.numbersImage

    def get_numbers_memory(self):
        '''
         Получить номера из памяти
        '''
        if self.memoryNumbers == None:
            return None

        if len(self.memoryNumbers[0]) < self.config.memory_number_frames:
            return None

        res = []
        for mem in self.memoryNumbers:
            cands = []
            counts = []
            for m in mem:
                if m == None or m == []:
                    continue
                for n in m:
                    if n[0] not in cands:
                        cands.append(n[0])
                        counts.append(1)
                    else:
                        i = cands.index(n[0])
                        counts[i] += 1

            numres = []
            if cands != []:                
            
                for i,c in enumerate(cands):
                    if counts[i] >= self.config.memory_number_repeat:                    
                        numres.append(c)

            res.append(numres)

        return res


    def draw_symbols(self,images,filenames,param = 1):
        '''
         Функция отладочного вывода детектированного номера в файл
         images- список изображений openCV, на котором нужно рисовать;
         filenames - список имен файлов.
         param  - 0, выводить символы отдельно
                  1, выводить номер целиком
        '''
        if self.list_objects is None:
            return None

        if len(images) != len(self.list_objects[0][0]):
            return None

        for symb in range(self.config.types_of_object_detection):
            boxes,dscores = self.list_objects[symb][0],self.list_objects[symb][1]        

            for j,img in enumerate(images):
                if boxes[j] is None:
                    continue

                w = img.shape[1]
            
                all_boxes,_ = boxes[j].shape
                font = cv2.FONT_HERSHEY_SIMPLEX            

                if param == 0:
                    for i in range(all_boxes):                    
                        x1 = boxes[j][i][1]
                        y1 = boxes[j][i][0]
                        x2 = boxes[j][i][3]
                        y2 = boxes[j][i][2]
        
                        img = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255),2)
                        s = '{}'.format(symb)
                    
                        yt = int(y1)-10
                        if yt < 30:
                            yt = int(y2)+30
                        textSize = cv2.getTextSize(s, fontFace=font, fontScale=1, thickness=2)
                        if x1 + textSize[0][0] > w:
                            x1 = w - textSize[0][0]
                        img = cv2.putText(img,s,(int(x1),yt),font,1,(0,0,255),2)

                if param == 1 and not self.numbersImage is None:
                    for n in self.numbersImage[j]:
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

        if filenames != None:
            for j,img in enumerate(images):
                cv2.imwrite(filenames[j],img)
        else:
            return images

        return None