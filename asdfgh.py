import numpy as np
import glob
import cv2
#from PIL import Image
from keras.preprocessing import image

path = glob.glob('/home/raghu/Desktop/withgpu/flickr/Flickr8k_Dataset/Flicker8k_Dataset/*.jpg')
images = list()
for img in path:
    #print(img)
    n = cv2.imread(img)
    #print(n)
    x = image.img_to_array(n)
    #print(x)
    x = np.resize(x, (299, 299))
    #print(x.shape)
    x = np.expand_dims(x, axis = 0)
    #print(x.shape)
    images.append(x)
    #x = x.reshape(1, 299, 299, 3)
    #x = np.expand_dims(x, axis = 3)
    #x = preprocess_input(x)