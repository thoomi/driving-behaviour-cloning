# **Behavioral Cloning**

This is the third project I am doing as part of Udacity's Self-Driving-Car Nanodegree.

**The goals / steps of this project are the following:**
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

# Report
### Writeup & Project Files

#### 1. Writeup
You're reading it! I will examine below the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.


#### 2. Project Files

You can find all project files in this [Github Repository](https://github.com/thoomi/driving-behaviour-cloning).

My project includes the following files:
* [model.py](https://github.com/thoomi/driving-behaviour-cloning/blob/master/model.py) containing the script to create and train the model
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

...

#### 2. Attempts to reduce overfitting in the model

...

#### 3. Model parameter tuning

...

#### 4. Appropriate training data

...

---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

...

#### 2. Final Model Architecture

...

#### 3. Creation of the Training Set & Training Process

...

---

### Appendix

#### Papers
[paper01]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
[paper02]: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
[paper03]: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
[paper04]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

- [End to End Learning for Self-Driving Cars [Bojarski et al. 2016]][paper01]


#### Blogs & Tutorials


#### Tools:
[tool01]: https://www.tensorflow.org/
[tool02]: http://matplotlib.org/
[tool03]: http://opencv.org/

 - [TensorFlow][tool01]
 - [Matplotlib][tool02]
 - [OpenCV][tool03]
