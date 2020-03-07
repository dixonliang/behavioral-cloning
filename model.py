import csv
import cv2
import numpy as np
from scipy import ndimage

lines = []
with open('./data_test/driving_log.csv') as csvfile: #import images based on csv file
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines[1:len(lines)]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data_test/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
augmented_images, augmented_measurements = [], []     # augment / flip images to rid of left steer bias 
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) # crop frames to save time in training the model
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) # normalize data
model.add(Convolution2D(6,5,5,activation="relu")) # layer 1
model.add(MaxPooling2D())
model.add(Dropout(0.2)) # add dropout layer after layer 1 to prevent overfitting of 20% dropout
model.add(Convolution2D(6,5,5,activation="relu")) # layer 2
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu")) # layer 3
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') # use mse to determine error loss, use adam optimizer for model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5) # validation split set of 20%, use 5 epochs

model.save('model.h5')