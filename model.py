"""Udacity P3 - Behavioral Cloning"""
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# =============================================================================
# Extract data from csv file and load all training images
# =============================================================================
recorded_data_path = './data/recorded/'
recorded_log_file = recorded_data_path + 'driving_log.csv'
recorded_image_path = recorded_data_path + 'IMG/'

# Load training data
images = []
measurements = []


def load_image(path):
    """Load image from local path"""
    filename = path.split('/')[-1]
    current_path = recorded_image_path + filename
    return cv2.imread(current_path)


with open(recorded_log_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # Create adjusted steering measurements for the side camera images
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # Read in images from center, left and right cameras
        img_center = load_image(row[0])
        img_left = load_image(row[1])
        img_right = load_image(row[2])

        # Add images and angles to data set
        images += [img_center, img_left, img_right]
        measurements += [steering_center, steering_left, steering_right]


# =============================================================================
# Augment training data
# =============================================================================
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    # Keep original image
    augmented_images.append(image)
    augmented_measurements.append(measurement)

    # Add flipped image and corresponding measurement
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

# Convert training data to np array
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# =============================================================================
# Build the neural network model
# =============================================================================
input_img_shape = (160, 320, 3)
validation_split = 0.2
optimizer = 'adam'
loss_function = 'mse'
epochs = 10
batch_size = 100

# Build model
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_img_shape))
model.add(Lambda(lambda x: x / 255.0 - 0.5))

model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss=loss_function, optimizer=optimizer)


# =============================================================================
# Train the model and output metrics
# =============================================================================
history = model.fit(X_train, y_train, validation_split=validation_split,
                    shuffle=True, nb_epoch=epochs, batch_size=batch_size)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# =============================================================================
# Save the final model and weights
# =============================================================================
model.save('model.h5')
print('Model saved.')
