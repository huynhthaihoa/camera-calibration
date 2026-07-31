# Camera Calibration Tool

## Introduction
To understand what is "camera calibration", I recommend reading the following articles before going on further steps:
- [MathWorks - What Is Camera Calibration?](https://www.mathworks.com/help/vision/ug/camera-calibration.html?lang=en)
- [Towards Data Science - What are Intrinsic and Extrinsic Camera Parameters in Computer Vision?](https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec)

This library is for **camera calibration** (estimating camera intrinsic parameters including **camera matrix** and **distortion coefficients**) and **undistortion** (removing **distortion effect**). 

## Supported camera models
- Fisheye camera
- Pinhole camera (normal camera)

## Supported sensor types
- RGB sensor
- Thermal (infrared) sensor

# Library structure
The library includes 2 scripts:
- [calibration.py](calibration.py): script to **calibrate** the camera/video file
- [undistortion.py](undistortion.py): script to **undistort** the camera/video file
- [calibration_images.py](calibration_images.py): script to **calibrate** the images (all images must be captured by the same camera)
- [undistortion_images.py](undistortion_images.py): script to **undistort** the images

Utilities:
- [utils.py](calibrator.py): Calibrator class and other utility functions
- [Checkerboard-A3-55mm-6x4.pdf](Checkerboard-A3-55mm-6x4.pdf): checkerboard pattern file for calibrating the camera
- [environment.yaml](environment.yaml): Anaconda environment file for installing dependencies

# Installation
- Create a new environment using Anaconda:
```conda env create -f environment.yaml```
- Activate the environment:
```conda activate calib```

# Usage

## 1. Calibration
### Disclaimer
The distortion coefficient values are in format k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, tx, ty]]]]) of 4, 5, 8, 12, or 14 elements:
- k[] values are the **radial coefficients**.
- p[] values are the **tangential distortion coefficients**.
- s[] values are the **thin prism distortion coefficients**.
- Higher-order coefficients are currently not considered. 

### 1.1. Calibration using stream data (video/USB stream)

1. Print the checkerboard and attach it on a rigid, flat surface (you can use the [existing file](Checkerboard-A3-55mm-6x4.pdf) or download from [here](https://markhedleyjones.com/projects/calibration-checkerboard-collection)). Make sure that there is a thick white border all around the checkerboard pattern. This white border is needed to facilitate corner detection.

2. Run the script **calibration.py** with the below command:

```
python calibration.py -i [index] -f [fisheye] -t [thermal] -l [low_res] -c [checkerboard_size] -p [param_thres] -q [quantity_thres] -o [output_path] -g [good] -d [detect]
```

where:

- **index**: input source (camera index/video path). The default value is `0`.
 - **fisheye**: if set, the camera is supposed to be a **fisheye model** (if it is supposed to be a **pinhole model**, one can skip it)
 - **thermal**: if set, the camera sensor is supposed to be a **thermal sensor** (if it is supposed to be an **RGB sensor**, one can skip it)
  - **low_res**: flag to set if the camera/video has a low resolution. The default value is `False` (if the camera/video has a low resolution, put flag `-l` into the command, otherwise, just skip it). If the camera/video has a considerably good resolution, it is highly recommended to skip this flag to achieve a better checkerboard detection result => better calibration result.
 - **checkerboard_size**: checkerboard size (the number of corners per row x the number of corners per column). The default value is `6, 4`.
 - **param_thres**: The parameter threshold for evaluating the detected corners is **good** and will be used for calibration. The default value is `0.2`.
 - **quantity_thres**: the minimum number of images with **good** detected corners that will be considered enough for calibration. The default value is `40` (images).
 - **output_path**: output calibration file path (xml file). The default path is `output.xml`.
 - **good**: directory to save **good** images (images that have a **good** detected corner). Leave it empty if one doesn't want to save.
 - **detect**: directory to save **detected** images (good corner detection result). Leave it empty if one doesn't want to save.
 
example:
```
python calibration.py -i 0 -f -l -c "10,7" -o "C:\Users\hoa\Pictures\Camera Roll\calibration.xml" -g good -d detect
```

3. Move the camera to capture the checkerboard image in different viewpoints. Try to cover all the visible areas of the camera and make sure that **every corner of the checkerboard is visible** from the camera's viewpoint. On each viewpoint where checkerboard corners are detected, these corners will be displayed:
```
  - If the script didn't detect corners in the current image, a message "Detect corners failed. Accumulated [number of collected images[] images!" will be displayed
  
  - If the script detected corners in the current image, but the detected corners aren't "good" enough to be used for the calibration, a message "Detect bad corners. Accumulated [number of collected images[] images!" will be displayed

  - If the script detected corners in the current image and the detected corners are "good" so that the image can be used for the calibration, a message "Detect good corners. Accumulated [number of collected images[] images!" will be displayed
    
  - If there are enough valid images for calibration, a message "Enough images for calibration!" will be displayed
```

To calibrate the thermal (infrared) camera, the checkerboard should be put on a hot surface(for example: under sunlight, under halogen light, etc.,) to enhance the contrast between "black" squares and "white" squares, which is essential for checkerboard detection algorithm. 

4. When the script has already collected enough valid images for calibration, press the button `Q` on the keyboard to finish inspecting images, or it will automatically finish if the input is a video and all frames have been inspected.
5. The calibration information will be displayed on the terminal, includes:
    - Number of "valid" images from the input stream that are used for calibration
    - **DIM**: input resolution
    - **K**: camera matrix
    - **D**: distortion coefficients
    - **rvecs**: rotation matrices estimated from each of the "valid" images
    - **tvecs**: translation vectors estimated from each of the "valid" images
    - **reproj_err**: reprojection error to judge the calibration performance, we aim
    to minimize this value as much as possible

### 1.2. Calibration using captured images

1. Run the script **calibration_images.py** with the below command:

```
python calibration_images.py -i [input] -f [fisheye] -t [thermal] -l [low_res] -c [checkerboard_size] -p [param_thres] -q [quantity_thres] -o [output_path] -g [good] -d [detect]
```

where:

- **input**: the directory which contains captured images.
 - **fisheye**: if set, the camera is supposed to be a **fisheye model** (if it is supposed to be a **pinhole model**, one can skip it)
 - **thermal**: if set, the camera sensor is supposed to be a **thermal sensor** (if it is supposed to be an **RGB sensor**, one can skip it)
  - **low_res**: flag to set if the camera/video has a low resolution. The default value is `False` (if the camera/video has a low resolution, put flag `-l` into the command, otherwise, just skip it). If the camera/video has a considerably good resolution, it is highly recommended to skip this flag to achieve a better checkerboard detection result => better calibration result.
 - **checkerboard_size**: checkerboard size (the number of corners per row x the number of corners per column). The default value is `6, 4`.
 - **param_thres**: The parameter threshold for evaluating the detected corners is **good** and will be used for calibration. The default value is `0.2`.
 - **quantity_thres**: the minimum number of images with **good** detected corners that will be considered enough for calibration. The default value is `40` (images).
 - **output_path**: output calibration file path (xml file). The default path is `output.xml`.
 - **good**: directory to save **good** images (images that have a **good** detected corner). Leave it empty if one doesn't want to save.
 - **detect**: directory to save **detected** images (good corner detection result). Leave it empty if one doesn't want to save.
 
example:
```
python calibration_images.py -i input -f -l -c "10,7" -o "C:\Users\hoa\Pictures\Camera Roll\calibration.xml" -g good -d detect
```

3. On each image where checkerboard corners are detected, these corners will be displayed:
```
  - If the script didn't detect corners in the current image, a message "Detect corners failed. Accumulated [number of collected images[] images!" will be displayed
  
  - If the script detected corners in the current image, but the detected corners aren't "good" enough to be used for the calibration, a message "Detect bad corners. Accumulated [number of collected images[] images!" will be displayed

  - If the script detected corners in the current image and the detected corners are "good" so that the image can be used for the calibration, a message "Detect good corners. Accumulated [number of collected images[] images!" will be displayed
    
  - If there are enough valid images for calibration, a message "Enough images for calibration!" will be displayed
```

To calibrate the thermal (infrared) camera, the checkerboard must be put into the hot surface. 

4. When the script has already collected enough valid images for calibration, press the button `Q` on the keyboard to finish inspecting images, or it will automatically finish if all images have been inspected.
5. The calibration information will be displayed on the terminal.

## 2. Undistortion

### Disclaimer 
Currently, the library cannot fully undistort the fisheye image with an angle of view >= 180: the region around the image border will be cropped out even if you set the zoom value as 1.

### 2.1. Undistortion on stream data (video/USB stream)
1. Run the script **undistortion.py** with the below command:

```
python undistortion.py -i [index] -c [calibration_path] -s [show_original] -z [zoom]
```
where:
- **index**: input source (camera index/video path). The default value is 0.
- **calibration_path**: calibration file path (xml file).
- **show_original**: show original video or not (if you don't want to show original video, just skip this param)
- **zoom**: Zoom value, ranges from `0` (zoomed in, all pixels in the calibrated image are valid) to `1`, (zoomed out, all pixels in the original image are in the calibrated image). The default value is `0`.
 <!-- - **camera_model**: camera model (0: pinhole, 1: fisheye). The default value is 0 (pinhole camera). -->

example:
```
python undistortion.py -i 0 -c "C:\Users\hoa\Pictures\Camera Roll\calibration.xml" -s -z 1.0
```
2. Press the button `Q` on the keyboard to exit the stream, or it will automatically finish if the input is a video and all frames have been inspected.
### 2.2. Undistortion on images
1. Run the script **undistortion_images.py** with the below command:

```
python undistortion_images.py -i [input] -o [output] -c [calibration_path] -s [show_original] -z [zoom]
```
where:
- **input**: input images directory.
- **output**: directory to save undistortion result image.
- **calibration_path**: calibration file path (xml file).
- **show_original**: show the original image or not (if you don't want to show the original image, just skip this param)
- **zoom**: Zoom value, ranges from `0` (zoomed in, all pixels in the calibrated image are valid) to `1`, (zoomed out, all pixels in the original image are in the calibrated image). The default value is `0`.
 <!-- - **camera_model**: camera model (`0`: pinhole, `1`: fisheye). The default value is `0` (pinhole camera). -->

example:
```
python undistortion_images.py -i "input" -o "output" -c "C:\Users\hoa\Pictures\Camera Roll\calibration.xml" -s -z 1.0
```
2. Press the button `Q` on the keyboard to finish inspecting images, or it will automatically finish if all images have been inspected.

# To do
- Make a GUI like [ROS calibration library](https://github.com/ros-perception/image_pipeline/tree/noetic/camera_calibration) to visualize the calibration progress.
- Support wide-angle fisheye camera full undistortion.
- Support other calibration file formats (TXT, NPY, etc.)
- Support direct undistortion after calibrating.
- Support higher-order distortion coefficients.
# Credit
The library is mostly based on [ROS calibration library](https://github.com/ros-perception/image_pipeline/tree/noetic/camera_calibration) 
