"""Udacity P3 - Behavioral Cloning - Data and model visualizations"""
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data_samples

# =============================================================================
# Load data samples
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

x = (np.array(samples)[:, 3]).astype(np.float)
y = np.linspace(-1, 1)


# =============================================================================
# Plot steering angles in "time" dimension
# =============================================================================
fig, ax = plt.subplots(2, 1)

ax[0].plot(x)
ax[0].set_ylabel('Steering angle')


# =============================================================================
# Plot steering angles histogram
# =============================================================================
hist, bins = np.histogram(x, bins=20)

width = 1 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

ax[1].bar(center, hist, align='center', width=width)
ax[1].set_xlabel('Steering angle')

plt.show()
