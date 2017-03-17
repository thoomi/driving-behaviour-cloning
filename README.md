## Project: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This is the third project I am doing as part of Udacity's Self-Driving-Car Nanodegree.

### Project Goals
The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[image9]: ./examples/cropped_example.png "Cropped example"
[image10]: ./examples/flipped_example.png "Flipped example"


### Implementation
The [final model](https://github.com/thoomi/driving-behaviour-cloning/blob/master/model.py) architecture is basically the Nvidia architecture from their End to End Learning for Self-Driving Cars paper combined with an additional dropout layer right before the fully-connected layers. By utilizing this architecture combined with self-recorded training data and some augmentation I managed to train a neural network capable of driving without human interaction around track 1.

 The augmentation included cropping:

![Cropped example][image9]


And flipped images:

| Original                   | Flipped                     |
|:--------------------------:|:---------------------------:|
| ![Cropped example][image9] | ![Flipped example][image10] |
| -0.1°                       | 0.1°                       |


And as well an [on demand training data generator](https://github.com/thoomi/driving-behaviour-cloning/blob/master/generator.py).

### Results
After all the car was able to drive autonomously around track 1.

[image6]: ./examples/video.gif "Final result"
![Final result drive][image6]


For a more detailed insight on the project please see the full [Writeup / Report](https://github.com/thoomi/driving-behaviour-cloning/blob/master/writeup_report.md).
