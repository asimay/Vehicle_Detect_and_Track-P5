
## **Vehicle Detection Project**

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle.png
[image2]: ./output_images/non-vehicle.png 
[image3]: ./output_images/car_hog.png
[image4]: ./output_images/notcar_hog.png
[image5]: ./output_images/slide_window.png
[image6]: ./output_images/slide_window3.png
[image7]: ./output_images/slide_window2.png

[image8]: ./output_images/bboxes_and_heat1.png
[image9]: ./output_images/bboxes_and_heat2.png
[image10]: ./output_images/bboxes_and_heat3.png
[image11]: ./output_images/bboxes_and_heat4.png
[image12]: ./output_images/final.gif
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Final video:
![alt text][image12]

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the “Combine and Normalize Features” code cell of the IPython notebook .  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and use:

```python
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) 
hist_bins = 64 
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC() and GridSearchCV() method, and I used RGB->YCrCb color space transform, bin_spatial, and color histogram method, and Hog feature extraction to extract combined features  and train the classifier.

```python
parameters = {'C':[1, 10] } 
svr = LinearSVC()
svc = GridSearchCV(svr, parameters)
```

Here is my running result:

```python
Test Accuracy of svc =  0.9887
best params is:  {'C': 1}
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I compute the span of the region to be searched, and compute the number of pixels per step in x/y, also this need to compute the number of windows in x/y, and then I Initialize a list to append window positions into it, and loop through finding x and y window positions, finally this will return the list of windows, and this is called slide window search method.

```python
y_start_stop = [400, 656]
xy_overlap = (0.5, 0.5)
xy_window = (64, 64)
scale_list = [2.1, 1.6, 1.0，0.8]
```

scales is set to [2.1, 1.6, 1.0，0.8], and overlap is set to 0.5 after try to tune the parameters.

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I used YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

#### 3. Here are 3 frames and their corresponding heatmaps, and output of `scipy.ndimage.measurements.label()` on the integrated heatmap, and the resulting bounding boxes are drawn onto the images:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. images data is too large when train, it is easy to fail, after add one more memory, it success. the memory must be no less than 8G after testing.
2. slide window might be fail when car is become farthur, we must add the scale list to adapt this situation.
3. when 2 cars are close enough, it will failed to recognize the two car as two, it will recognize the two car as one car. this need to investigate further more.


