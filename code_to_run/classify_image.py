import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv
import argparse


ap = argparse.ArgumentParser()
# Image argument
ap.add_argument("-i", "--image", required=True, help='path to input image')
args = vars(ap.parse_args())

image_path  = args['image']
model = keras.models.load_model('/home/daxter-army/Desktop/ml_report/code_to_run/CURRENCY_MODEL')
class_names = ['10', '100', '20', '200']

img = keras.preprocessing.image.load_img(
    image_path, target_size=(180, 180)
)
    
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

print('==================Class Wise Scores================')
print('======10======')
print('{:.2f}'.format(100 * score[0]))
print('======100======')
print('{:.2f}'.format(100 * score[1]))
print('======20======')
print('{:.2f}'.format(100 * score[2]))
print('======200======')
print('{:.2f}'.format(100 * score[3]))