# iANRCRDetect - детектирование цифр на жд вагонах
import cv2
import iANRCRSettings
import tensorflow as tf
import numpy as np

from util.tf_load_fz import load_frozen_graph


class iANRCRDetect(object):
    def __init__(self,config):        
        self.config = config                
        self.detect_function = load_frozen_graph(iANRCRSettings.iANRCRDetectModelPath)        
            
    def process(self,im):
        '''
         Детектирование номеров
             im - массив изображений shape:
             (batch,h,w, ch)
        Возвращение:
            массивы boxes и scores
        '''                
        b, h, w,ch = im.shape  
        yout = self.detect_function(x=tf.constant(im))
        
        yout = [x if isinstance(x, np.ndarray) else x.numpy() for x in yout]
 
        y=yout[0]
        y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels   

        list_objects_n = None 
        images,_,_ = y.shape
        for k in range(images):
            yindexes = np.where(y[k][..., 4] > self.config.detect_conf_thresh)
            all_cands = len(yindexes[0])    
            if all_cands > 0:
                # Надо преобразовать в нужный формат для NMS tf для каждого класса отдельно
                box = np.zeros(4)                
                list_objects = []
                objects = []
                for i in range(self.config.types_of_object_detection):
                    list_objects.append([None,None]) # boxes,scores     
                    objects.append([[None],[None]])
                j = -1                            
                for i in yindexes[0]:                                                    
                    j += 1
                    if j >= all_cands: break                    
                    t = np.argmax(y[k][i][5:])
                    box[:] = [y[k][i][1] - y[k][i][3]/2,y[k][i][0] - y[k][i][2]/2,y[k][i][1] + y[k][i][3]/2,y[k][i][0] + y[k][i][2]/2]
                    if list_objects[t][0] is None:
                        list_objects[t][0] = np.zeros((1, 4))
                        list_objects[t][1] = np.zeros((1))                                        
                        list_objects[t][0][0][:] = box                    
                        list_objects[t][1][0] = y[k][i][4]                        
                    else:
                        list_objects[t][0] = np.vstack([list_objects[t][0],box])
                        list_objects[t][1] = np.append(list_objects[t][1],y[k][i][4])                                    
                for t in range(self.config.types_of_object_detection):
                    if list_objects[t][0] is None:
                        continue
                    # NMS по классам
                    #Перевод в Tf tensor                
                    boxes_tf = tf.constant(list_objects[t][0],dtype=float)
                    scores_tf = tf.constant(list_objects[t][1],dtype=float)                            
                    #NMS
                    selected_indices,selected_scores = tf.image.non_max_suppression_with_scores(boxes_tf, scores_tf, self.config.detect_max_output_size)                        
                    selected_boxes = tf.gather(list_objects[t][0], selected_indices)                
                    # Результаты возвращаются как y1 x2 y2 x2 Пример:
                    #print(selected_boxes,selected_scores)        
                    # tf.Tensor([[350.82800293 174.18496704 400.8684082  280.00772095]], shape=(1, 4), dtype=float64)  
                    objects[t] = [[selected_boxes.numpy()],[selected_scores.numpy()]]
                if list_objects_n is None:
                    list_objects_n = []
                    for t in range(self.config.types_of_object_detection):
                        list_objects_n.append(objects[t]) 
                else:
                    for t in range(self.config.types_of_object_detection):
                        list_objects_n[t][0].append(objects[t][0][0])
                        list_objects_n[t][1].append(objects[t][1][0])
            else:
                if list_objects_n is None:
                    list_objects_n = []
                    for t in range(self.config.types_of_object_detection):
                        list_objects_n.append([[None],[None]])                  
                else:
                    for t in range(self.config.types_of_object_detection):
                        list_objects_n[t][0].append(None)
                        list_objects_n[t][1].append(None)

        return list_objects_n
