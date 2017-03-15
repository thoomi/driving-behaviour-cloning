# **Behavioral Cloning**

This is the third project I am doing as part of Udacity's Self-Driving-Car Nanodegree.

**The goals/steps of this project are the following:**
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/recover_from_left.gif "Recovery Example"
[image2]: ./examples/recover_from_right.gif "Recovery Example"
[image3]: ./examples/curve_left_after_bridge.gif "Curve Example"
[image4]: ./examples/curve_right_after_bridge.gif "Curve Example"
[image5]: ./examples/cnn-architecture.png "Network architecture"
[image6]: ./examples/video.gif "Final result"

# Report
### Writeup & Project Files

#### 1. Writeup
You're reading it! I will examine below the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.


#### 2. Project Files

You can find all project files in this [Github Repository](https://github.com/thoomi/driving-behaviour-cloning).

My project includes the following files:
* [model.py](https://github.com/thoomi/driving-behaviour-cloning/blob/master/model.py) containing the script to create and train the model
* [video.mp4](https://github.com/thoomi/driving-behaviour-cloning/blob/master/video.mp4) record of the model driving track 1 successfully
* [drive.py](https://github.com/thoomi/driving-behaviour-cloning/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/thoomi/driving-behaviour-cloning/blob/master/model.h5) containing a trained convolution neural network
* [writeup_report.md](https://github.com/thoomi/driving-behaviour-cloning/blob/master/writeup_report.md)

---

### Quality of Code

#### 1. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```


#### 2. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The final model consists
(model.py lines 143-163)

#### 2. Attempts to reduce overfitting in the model

In order to prevent the model from overfitting, I introduced a dropout layer (model.py line 156) with a keep probability of 50% after the convolutional layers. This helps the model to generalize by not relying too much on every extracted feature. A second attempted to reduce overfitting, was to limit the training epochs to the range between 3-6 (model.py line 22).
To validate the model, I used only 1% of the initial data because, in this project, a low validation loss does not necessarily mean a good performance on the track.

#### 3. Model parameter tuning

I used the Adam optimizer with the default Keras settings. Additionally, I set the training epochs to be around 5 and a batch size of 200. Other parameters have been tested, but I figured them as working best in my case (model.py lines 18-23).

#### 4. Appropriate training data

My approach to collect the training data with the simulator was an iterative one. First, I collected two rounds of center line driving clockwise and two rounds of center line driving counter-clockwise respectively. Those four rounds served as my base data. After that, I collected two to three attempts to recover from each side of the lane. Seeing the model driving of the street in the left and right curve directly after the bridge, I collected some more data driving only those two curves. In a nutshell, my training looks like this:

* 2 x center line driving clockwise
* 2 x center line driving counter-clockwise
* 3 x attempts recovering from left
* 3 x attempts recovering from right
* 2 x driving the left curve after the bridge
* 2 x driving the right curve after the bridge

##### Training data examples

| Recover from left            |      Recover from right       |
|:----------------------------:|:-----------------------------:|
| ![Recover from left][image1] | ![Recover from right][image2] |


| Curve left after bridge            |      Curve right after bridge       |
|:----------------------------------:|:-----------------------------------:|
| ![Curve left after bridge][image3] | ![Curve right after bridge][image4] |



---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

![Network architecture][image5]

#### 2. Final Model Architecture

The final model architecture is basically the Nvidia architecture from their End to End Learning for Self-Driving Cars paper combined with an additional dropout layer right before the fully-connected layers.

#### 3. Creation of the Training Set & Training Process

I recorded the training data in steps. After each step, I trained and evaluated the model's performance by letting it drive in the simulators autonomous mode while observing its behavior.

---

### Simulation

 The car was able to drive endlessly on track 1 ;)

![Final result drive][image6]

---

### Appendix

#### Papers
[paper01]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
[paper04]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

- [End to End Learning for Self-Driving Cars [Bojarski et al. 2016]][paper01]


#### Blogs & Tutorials


#### Tools:
[tool01]: https://keras.io/
[tool02]: http://matplotlib.org/
[tool03]: http://opencv.org/

 - [Keras][tool01]
 - [Matplotlib][tool02]
 - [OpenCV][tool03]
