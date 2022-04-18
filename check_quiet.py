from skimage import io
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

batch_size = 32
img_height = 288
img_width = 352

# Image Stuff
image_url = "http://192.168.1.4:7080/api/2.0/snapshot/camera/5ba66da8e26c83b5037d63cf?apiKey=tzW1eq2ikQIoN2RshfnhjXu5JVoVj2UA&force=true&t=1516046678"
img_array = io.imread(image_url)
img_array = tflite.expand_dims(img_array, 0)


# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = img_array

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

#debug
print(output_data)

class_names = ['mercedes_none', 'mercedes_other', 'mercedes_tesla', 'mercedes_toyota', 'none_mercedes', 'none_none',
               'none_other', 'none_tesla', 'none_toyota', 'other_mercedes', 'other_none', 'other_tesla', 'other_toyota',
               'tesla_mercedes', 'tesla_none', 'tesla_other', 'tesla_toyota', 'toyota_mercedes', 'toyota_none',
               'toyota_other', 'toyota_tesla']


# prediction

save_stdout = sys.stdout
sys.stdout = open('/dev/null', 'w')



sys.stdout = save_stdout



predictions = model.predict(img_array)
score = tflite.nn.softmax(predictions[0])

class_names = ['mercedes_none', 'mercedes_other', 'mercedes_tesla', 'mercedes_toyota', 'none_mercedes', 'none_none',
               'none_other', 'none_tesla', 'none_toyota', 'other_mercedes', 'other_none', 'other_tesla', 'other_toyota',
               'tesla_mercedes', 'tesla_none', 'tesla_other', 'tesla_toyota', 'toyota_mercedes', 'toyota_none',
               'toyota_other', 'toyota_tesla']

os.remove(image_path)

result_group = class_names[np.argmax(score)]
result_score = 100 * np.max(score)
results = result_group.split("_")

print("{\"top\": \"" + results[0] + "\", \"bottom\": \"" + results[1] + "\", \"score\": \"" + str(result_score) + "\"}")
