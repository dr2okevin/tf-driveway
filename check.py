import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


batch_size = 32
img_height = 288
img_width = 352


model = tf.keras.models.load_model('trained_model')

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# prediction

image_url = "http://192.168.1.4:7080/api/2.0/snapshot/camera/5ba66da8e26c83b5037d63cf?apiKey=tzW1eq2ikQIoN2RshfnhjXu5JVoVj2UA&force=true&t=1516046678"
image_path = tf.keras.utils.get_file('auffahrt', origin=image_url)

img = keras.preprocessing.image.load_img(
    image_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_names = ['mercedes_none', 'mercedes_other', 'mercedes_tesla', 'mercedes_toyota', 'none_mercedes', 'none_none', 'none_other', 'none_tesla', 'none_toyota', 'other_mercedes', 'other_none', 'other_tesla', 'other_toyota', 'tesla_mercedes', 'tesla_none', 'tesla_other', 'tesla_toyota', 'toyota_mercedes', 'toyota_none', 'toyota_other', 'toyota_tesla']
 

print(
    "{} {:.2f} %"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    #.format(np.argmax(score), 100 * np.max(score))
)

os.remove(image_path)
