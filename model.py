"""Udacity P3 - Behavioral Cloning"""
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# =============================================================================
# Hyperparameters
# =============================================================================
input_img_shape = (160, 320, 3)
validation_split = 0.2
optimizer = 'adam'
loss_function = 'mse'
epochs = 5
batch_size = 50

# =============================================================================
# Extract data from csv file and load all training images
# =============================================================================
data_folders = ['forward_round_1',
                'forward_round_2',
                'backward_round_1',
                'backward_round_2']

# =============================================================================
# Extract data from csv file and load all training images
# =============================================================================
recorded_base_path = './data/recorded/'
recorded_log_file = 'driving_log.csv'
recorded_image_path = '/IMG/'

samples = []
for folder in data_folders:
    log_file = recorded_base_path + folder + '/' + recorded_log_file

    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[0] = recorded_base_path + folder + recorded_image_path + line[0].split('/')[-1]
            line[1] = recorded_base_path + folder + recorded_image_path + line[1].split('/')[-1]
            line[2] = recorded_base_path + folder + recorded_image_path + line[2].split('/')[-1]

            samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=validation_split)


def load_image(path):
    """Load a single image from the recorded folder."""
    return cv2.imread(path)


def generator(samples, batch_size=32):
    """Generate batch sized data to feed the neural network"""
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            # Load captured training data
            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                # Create adjusted steering measurements for the side camera images
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # Read in images from center, left and right cameras
                img_center = load_image(batch_sample[0])
                img_left = load_image(batch_sample[1])
                img_right = load_image(batch_sample[2])

                images += [img_center, img_left, img_right]
                angles += [steering_center, steering_left, steering_right]

            # Produce flipped images
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, angles):
                # Keep original image
                augmented_images.append(image)
                augmented_measurements.append(measurement)

                # Add flipped image and corresponding measurement
                # augmented_images.append(cv2.flip(image, 1))
                # augmented_measurements.append(measurement * -1.0)

            # Yield the data in batch sized junks
            num_augmented_samples = len(augmented_images)
            for chunk_offset in range(0, num_augmented_samples, batch_size):
                X_train_chunk = augmented_images[chunk_offset:chunk_offset + batch_size]
                y_train_chunk = augmented_measurements[chunk_offset:chunk_offset + batch_size]

                # Convert training data into numpy arrays
                X_train_chunk = np.array(X_train_chunk)
                y_train_chunk = np.array(y_train_chunk)

                yield shuffle(X_train_chunk, y_train_chunk)


train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# =============================================================================
# Build the neural network model
# =============================================================================
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
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 3,
                              validation_data=validation_generator,
                              nb_val_samples=len(validation_samples),
                              nb_epoch=epochs)

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
