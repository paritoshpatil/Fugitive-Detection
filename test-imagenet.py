#import the necessary packages
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import cv2
 

 
# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread("/images/inputs/team-2.jpg")

cv2.imshow(orig)