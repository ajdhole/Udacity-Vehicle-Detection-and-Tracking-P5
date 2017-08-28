# Udacity Self-Driving Car Engineer Nanodegree Program¶
## Vehicle Detection Project¶

![alt text][image8]

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/data_visualization.png "Data Visualization"
[image2]: ./writeup_images/Hog_Display.png "Hog Display"
[image3]: ./writeup_images/initial_detection.png "Window search"
[image4]: ./writeup_images/heat_images.png "Window search without heatmap"
[image5]: ./writeup_images/heat_map_apply.png "Window search with heatmap"
[image6]: ./writeup_images/labels_map.png
[image7]: ./writeup_images/output_bboxes.png
[image8]: ./writeup_images/Comboned_Vehicle_Lane_Tracking.gif "Combined_Image"

[video1]: ./writeup_images.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 5th code cell of the IPython notebook of the file called `Vehicle-Detection-and-Tracking-P5.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented with a number of different combinations of color spaces and HOG parameters and trained a  SVM classifier using different combinations of HOG features extracted from the color channels. based on different combination of parameter following paramer]ter gives more accuracy for classifier and stable result. 

`color_space` = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb 

`orient` = 10  # HOG orientations

`pix_per_cell` = 8 # HOG pixels per cell

`cell_per_block` = 2 # HOG cells per block

`hog_channel` = 'ALL' # Can be 0, 1, 2, or "ALL" 

`spatial_size` = (32, 32) # Spatial binning dimensions

`hist_bins = 64`    # Number of histogram bins

`spatial_feat` = True # Spatial features on or off

`hist_feat` = True # Histogram features on or off

`hog_feat` = True # HOG features on or off`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Initially, I trained model with a linear SVM classifier, later to fine tune classifier parameter I used `GridSearchCV` parameter optimization function to select best parameter for classification, best parameter displayed with `clf.best_params_` and it is selected as  `{'kernel': 'rbf', 'C': 10}` and Test Accuracy of SVC yields  99.63%.


###Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used `slide_window` function to define sliding window search, Initially search area was complete image later it is restricted to `y_start_stop=[400, 656]`, The result of sliding search windows as shown below:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The final classifier uses scales and HOG features from all 3 channels of images in `YCrCb` space. The feature vector contains also spatially binned color and histograms of color features, The false positives were filtered out by using a heatmap approach as described below. Here are some typical examples of detections:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to my video result.

[![Track 2 Video](https://img.youtube.com/vi/O00Lt-0B39M/0.jpg)](https://youtu.be/O00Lt-0B39M)


#### 2 . Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I defined a heatmap function
`heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw ] +=1`
and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried above Vehicle detection pipeline on harder challange video, It seems it is failing to detect vehicle coming from opposite direction, since we train model with back side images of car only so, to make it more robust vehicle training data to be enhanced like vehicle images from all side, also some false detection occured at tree shadow, further it can be optimised by adapting different classification approach like decision tree or with convolutional neural network aproach. 
