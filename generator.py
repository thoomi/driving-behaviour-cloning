"""Udacity P3 - Behavioral Cloning - Training data generator"""
import cv2
import numpy as np
import random

from sklearn.utils import shuffle


def load_image(path):
    """Load a single image from the recorded folder."""
    return cv2.imread(path)


def train_data_generator(samples, batch_size=32):
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

                if (steering_center < 0.2 and steering_center > -0.2 and random.random() < 0.7):
                    continue

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
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            # Yield the data in batch sized junks
            num_augmented_samples = len(augmented_images)
            for chunk_offset in range(0, num_augmented_samples, batch_size):
                X_train_chunk = augmented_images[chunk_offset:chunk_offset + batch_size]
                y_train_chunk = augmented_measurements[chunk_offset:chunk_offset + batch_size]

                # Convert training data into numpy arrays
                X_train_chunk = np.array(X_train_chunk)
                y_train_chunk = np.array(y_train_chunk)

                yield shuffle(X_train_chunk, y_train_chunk)


def validation_data_generator(samples, batch_size=32):
    """Generate batch sized data to validate the neural network"""
    while 1:
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_center = load_image(batch_sample[0])
                steering_center = float(batch_sample[3])

                images.append(img_center)
                angles.append(steering_center)

            X_batch = np.array(images)
            y_batch = np.array(angles)

            yield (X_batch, y_batch)
