'''
 Загрузка tf frozen graph
 Взято из Yolo5
'''
import tensorflow as tf
import numpy as np

def load_frozen_graph(path):
    '''
     Функция возвращает wrapper
    '''
    w = path
    gd = tf.Graph().as_graph_def()  # TF GraphDef
    with open(w, 'rb') as f:
        gd.ParseFromString(f.read()) 

    # Входной и выходные слои
    inputLayer = "x:0"
    name, input = [], []
    for n in gd.node:  
        name.append(n.name)
        input.extend(n.input)
    outputLayers = sorted(f'{x}:0' for x in list(set(name) - set(input)) if not x.startswith('NoOp'))

    def wrapper_function(gd, inputs, outputs):
        x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
        ge = x.graph.as_graph_element
        return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))
        
    return wrapper_function(gd, inputs="x:0", outputs=outputLayers)
