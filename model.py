"""Udacity P3 - Behavioral Cloning"""
from data_loader import load_data_samples
from generator import train_data_generator
from generator import validation_data_generator
from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split


# =============================================================================
# Hyperparameters
# =============================================================================
input_img_shape = (160, 320, 3)
validation_split = 0.01
optimizer = 'adam'
loss_function = 'mse'
epochs = 5
batch_size = 200


# =============================================================================
# Extract data from csv file and load all training images
# =============================================================================
data_folders = ['forward_round_1',
                'forward_round_2',
                'backward_round_1',
                'backward_round_2',
                'recover_from_left',
                'recover_from_right',
                'curve_left_after_bridge',
                'curve_right_after_bridge']

samples = load_data_samples(data_folders)

train_samples, validation_samples = train_test_split(samples, test_size=validation_split)

train_generator = train_data_generator(train_samples, batch_size=batch_size)
validation_generator = validation_data_generator(validation_samples, batch_size=batch_size)


# =============================================================================
# Build the neural network model
# =============================================================================
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_img_shape))
model.add(Lambda(lambda x: x / 255.0 - 0.5))

model.add(Convolution2D(3, 1, 1, activation='relu', subsample=(1, 1)))
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))

model.add(Flatten())

model.add(Dropout(0.7))

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss=loss_function, optimizer=optimizer)


# =============================================================================
# Train the model and output metrics
# =============================================================================
history = model.fit_generator(train_generator, samples_per_epoch=batch_size * 100,
                              validation_data=validation_generator,
                              nb_val_samples=len(validation_samples),
                              nb_epoch=epochs)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()


# =============================================================================
# Save the final model and weights
# =============================================================================
model.save('model.h5')
print('Model saved.')
