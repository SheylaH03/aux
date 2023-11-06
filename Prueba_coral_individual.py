import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import numpy as np
import datetime

hora_actual = datetime.datetime.now
t = hora_actual()
T = []

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model.tflite')
label_file = os.path.join(script_dir, 'labels.txt')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

size = common.input_size(interpreter)

image = Image.open("../conjunto_datos/prueba/cats/cat.4276.jpg").convert('RGB').resize(size, Image.ANTIALIAS)

common.set_input(interpreter, image)
for u in range(10):
    t = hora_actual()
    interpreter.invoke()
    T += [(hora_actual() - t).total_seconds()]

    classes = classify.get_classes(interpreter, top_k=1)
    print('%.1fms' % (T[u] * 1000))

labels = dataset.read_label_file(label_file)
print(classes[0].id)
for c in classes:
    print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

print('Suma de tiempos de inferencia: %.1fms' % (sum(T)*1000))
print('Media de tiempos de inferencia: %.1fms' % (np.mean(T)*1000))
print('Desviacion estandar de tiempos de inferencia: %.1fms' % (np.std(T)*1000))
