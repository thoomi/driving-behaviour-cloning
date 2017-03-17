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
[image5]: ./examples/model_thumbnail.png "Network architecture"
[image6]: ./examples/video.gif "Final result"
[image7]: ./examples/steering_angle_histogram.png "Steering angle plot 1"
[image8]: ./examples/steering_angle_per_datapoint.png "Steering angle plot 2"
[image9]: ./examples/cropped_example.png "Cropped example"
[image10]: ./examples/flipped_example.png "Flipped example"
[image11]: ./examples/center_cropped_example.png "Center example"
[image12]: ./examples/left_cropped_example.png "Left example"
[image13]: ./examples/right_cropped_example.png "Right example"

# Report
### Writeup & Project Files

#### 1. Writeup
You're reading it! I will examine below the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.


#### 2. Project Files

You can find all project files in this [Github Repository](https://github.com/thoomi/driving-behaviour-cloning).

My project includes the following files:
* [model.py](https://github.com/thoomi/driving-behaviour-cloning/blob/master/model.py) containing the script to create and train the model
* [generator.py](https://github.com/thoomi/driving-behaviour-cloning/blob/master/generator.py) containing the script to preprocess the data
* [data_loader.py](https://github.com/thoomi/driving-behaviour-cloning/blob/master/generator.py) containing the script to load the recorded data from multiple folders
* [visualizations.py](https://github.com/thoomi/driving-behaviour-cloning/blob/master/visualizations.py) containing the script to create some training data visualizations
* [drive.py](https://github.com/thoomi/driving-behaviour-cloning/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/thoomi/driving-behaviour-cloning/blob/master/model.h5) containing a trained convolution neural network
* [video.mp4](https://github.com/thoomi/driving-behaviour-cloning/blob/master/video.mp4) record of the model driving track 1 successfully
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

In the generator.py file lives the code of a generator which preprocesses and augments the initial training data. It yields the processed data in batch-sized chunks.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After iterating on the LeNet-5 model I settled with a slightly modified [Nvidia architecture][paper01]. Please see section "Model Architecture and Training Strategy" for a full explanation.

#### 2. Attempts to reduce overfitting in the model

In order to prevent the model from overfitting, I introduced a dropout layer (model.py line 58) with a keep probability of 70% after the convolutional layers. This helps the model to generalize by not relying too much on every extracted feature. A second attempt to reduce overfitting was to limit the training epochs to 5 (model.py line 18).
To validate the model, I used only 1% of the initial data because, in this project, a low validation loss does not necessarily mean a good performance on the track.

#### 3. Model parameter tuning

I used the Adam optimizer with the default Keras settings. Additionally, I set the training epochs to be around 5 and a batch size of 200. Other parameters have been tested, but I figured them as working best in my case (model.py lines 14-19).

#### 4. Appropriate training data

My approach to collect the training data with the simulator was an iterative one. First, I collected two rounds of center line driving clockwise and two rounds of center line driving counter-clockwise respectively. Those four rounds served as my base data. After that, I collected two to three attempts to recover from each side of the lane. Seeing the model driving of the street in the left and right curve directly after the bridge, I collected some more data driving only those two curves. In a nutshell, my training looks like this:

* 2 x center line driving clockwise
* 2 x center line driving counter-clockwise
* 3 x recovering from left
* 3 x recovering from right
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

I started off with the standard and unmodified LeNet-5 model with the aim to record appropriate training data. As already stated, I recorded four full rounds of driving track 1 (two clockwise and two counter-clockwise) as the basic starting point. Afterward, I trained and evaluated the model's performance by letting it drive in the simulators autonomous mode while observing its behavior. The next step was to look where the car went off the drivable ground and explicitly record data only for those specific sections.

After countless attempts to record appropriate training data for the LeNet-5 model and failing all the time, I decided to switch my neural network to the Nvidia architecture. The new architecture proofed to be a very good choice for my recorded data. Even the first training session resulted in a fully driven round! Motivated by this success, I adjusted the model slightly, added some data augmentation, and as a result, the model learned even better and faster with fewer data.


#### 2. Final Model Architecture
[![Network architecture][image5]]( ./examples/model.png?raw=true)


The final model architecture is basically the Nvidia architecture from their [End to End Learning for Self-Driving Cars][paper01] paper combined with an additional dropout layer right before the fully-connected layers. Furthermore, I added a 1x1x3 convolution at the beginning of the network to let it figure out the best color options by itself. [[Paper04][paper04]]

The code for the defined model exists in mode.py between line numbers 45-65.

#### 3. Creation of the Training Set & Training Process

As stated in section "4. Appropriate training data", I collected training data iteratively while controlling the steering angle by mouse and driving at a speed of around 10 mph.

![Steering angle per datapoint][image8]
The image above shows the steering angle for all recorded data frames. You can view the x-axis as a time dimension. The peaks correspond to a higher steering angle while driving the curves on track 1.


![Steering angle histogram][image7]
The histogram shows the steering angles fitted into 20 bins. As we can clearly see, there is a very high rate of steering angles close to zero. Training the network on this data without any augmentation could lead to a model which will very likely drive straight off curves. But by just leaving out those small steering angles we might get a model which drives not very smooth in non-curved areas and zick-zacking off the path. My Solution to tackle this conflict was to randomly exclude 50% of images with a steering angle between -0.1 and 0.1.


![Cropped example][image9]

The image above shows a cropped version of input images (model.py line 46). I do this in order to prevent the model from being distracted of to much background.


| Original                   | Flipped                     |
|:--------------------------:|:---------------------------:|
| ![Cropped example][image9] | ![Flipped example][image10] |
| -0.1°                       | 0.1°                       |

Additionally, the images are flipped vertically and the steering angle is multiplied by -1 (generator.py lines 52-53). This process should prevent the model from steering too much in one direction.



| Left                     | Center                     | Right                     |
|:------------------------:|:--------------------------:|:-------------------------:|
| ![Left example][image12] | ![Center example][image11] | ![Right example][image13] |
| 0.2°                    | 0°                         | -0.2°                      |

Furthermore, I used the left and right camera images with a static steering correction factor of +-0.2  (generator.py lines 32-42). By including those images in the training process the network learns how to steer back from off-center locations on the track.

---

### Simulation

The car was able to drive endlessly on track 1. There is a [video](https://github.com/thoomi/driving-behaviour-cloning/blob/master/video.mp4?raw=true) containing a recorded of a full autonomous drive for one round.

![Final result drive][image6]

---


### Reflections & Further Work

I really enjoyed doing this project. The feeling I had while seeing the car drive autonomously for a full round was totally worth the effort. It began with quite frustrating iterations on the training data and LeNet-5 model. But seeing the car not going off the street after switching to the Nvidia architecture was awesome.

It is a bit sad to stop here for now because there, of course, is track 2. This one seems much harder to drive than track 1. If there is some time left after finishing the other projects I will definitely come back and try to train a network for track 2.

---

### Appendix

#### Papers
[paper01]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
[paper02]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
[paper03]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
[paper04]: https://arxiv.org/pdf/1511.01064.pdf

- [End to End Learning for Self-Driving Cars [Bojarski et al., 2016]][paper01]
- [Gradient-Based Learning Applied to Document Recognition [LeCun et al., 1998]][paper02]
- [Dropout:  A Simple Way to Prevent Neural Networks from
Overfitting [Srivastava et al., 2014]][paper03]
- [Color Space Transformation Network [Karargyris, 2015]][paper04]


#### Tools:
[tool01]: https://keras.io/
[tool02]: http://matplotlib.org/
[tool03]: http://opencv.org/

 - [Keras][tool01]
 - [Matplotlib][tool02]
 - [OpenCV][tool03]
