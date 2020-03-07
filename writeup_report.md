# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/curve.jpg "curve"
[image2]: ./examples/dirt.jpg "dirt"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a three layer convolution neural network with 6x5 filter sizes followed by a pooling layer after each one. The model includes RELU layers to introduce nonlinearity (model.py code lines 42, 45, 47), and the data is normalized in the model using a Keras lambda layer (code line 41). Each driving frame was cropped to train the model faster. 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer after the first convolutuon layer in order to reduce overfitting (code line 44). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 55). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line line 54).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving parts of the track in reverse. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a basic model and build from the ground up based on testing on the simulator on how well the car was driving. After trying a basic regression model, it was obvious that I needed to use something more complex. My first step was to use a convolution neural network model. 

In order to gauge how well the model was working, I split my image and steering angle data into a training (80% of the data) and validation set (20% of the data). To combat overfitting, I modified the model so that there was a dropout layer of 20% after the first convolutional layer. 

The final step was to run the simulator to see how well the car was driving around track one. There were two spots where the car seemed to drive off the edge (as seen per below). The initial models seemed to drift immediately after curves and couldn't recongize that the dirt boundaries were in fact boundaries. To improve the driving behavior in these cases, I had to retrain by gathering more data specifically those instances including recovery driving if we got close to the boundaries (sharp turning away from the boundary). 

<img src="./examples/curve.jpg" alt="alt text" width=200 height=200>
<img src="./examples/dirt.jpg" alt="alt text" width=200 height=200>

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (code lines 39-52) consisted of a convolution neural network with three 6x5 convolutional layers each with Relu activation, three pooling layers after each convolutonal layer, and one dropout layer. The model was also fed images that were cropped for faster training, augmented/flipped so that steering bias were removed, and normalized. 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. To augment the data set, I also flipped images and angles thinking that this would help deal with the bias of steering left in track one. I also went through parts of the track in reverse which was effectively new data points for the model. Something that was immediately evident as mentioned earlier was that the initial training data did not include examples for what to do if the car veered of the track. As a result, I needed to record several instances through the track of steering back to the center when the car found itself near the edges for one reason or another. 

After the collection process, I had about 40,000 data points. I then preprocessed this data by cropping the top portions of images capture that were not relevant for driving (ex. trees, parts of the sky, etc.). I also normalized the images using Keras. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was five as evidenced by the rate the error was decreasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
