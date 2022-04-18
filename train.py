# Load the TensorBoard notebook extension
import tensorboard

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import datetime
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

data_dir = pathlib.Path('/home/kevin/Bilder/nn-auffahrt-labeling/')

image_count = len(list(data_dir.glob('*/*.*')))
print(image_count)

# mercedes_none = list(data_dir.glob('mercedes_none/*'))
# PIL.Image.open(str(mercedes_none[0]))

# none_none = list(data_dir.glob('none_none/*'))
# PIL.Image.open(str(none_none[0]))

#batch_size = 32
batch_size = 16
img_height = 288
img_width = 352
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Number of Output Neuron == Number of image Groups
num_classes = 21

# model = tf.keras.models.load_model('trained_model')

# if model == false:
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Show summary of the model
model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# train the model
epochs = 20  # 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback]
)

# Save a normal Keras Model
model.save('trained_model')

# Save a TensorFlow Lite Model
# - Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# - Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# prediction

image_url = "http://192.168.1.4:7080/api/2.0/snapshot/camera/5ba66da8e26c83b5037d63cf?apiKey=tzW1eq2ikQIoN2RshfnhjXu5JVoVj2UA&force=true&t=1516046678"
image_path = tf.keras.utils.get_file('auffahrt', origin=image_url)

img = keras.preprocessing.image.load_img(
    image_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


result_group = class_names[np.argmax(score)]
result_score = 100 * np.max(score)
results = result_group.split("_")

print ("{\"top\": \""+results[0]+"\", \"bottom\": \""+results[1]+"\", \"score\": \""+str(result_score)+"\"}")
