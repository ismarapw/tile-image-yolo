# Tile Image for YOLO Format

## Problem
Sometimes small object in large resolution image need to be detected. If we try to labels the small object in a high resolution image and feed the data to the network with certain size (416x416 for example), then it will lose the detail of the small object because it is going to be compressed. As a result, the network cannot detect the object properly. 

We can try to set the network input size based on the original image resolution, but most of the network architecture for object detection has constraint input size. The larger input size used in the network the larger memory in GPU will be used for training process. This is not very efficient for certain condition especially for those who has GPU memory limitation. 

## Tile image
This repo try to solve the problem by using image pre-processing technique called tile images. This such type of image processing can divide an image to several chunks images based on the grid size (3x3, 5x5, etc). The script also divide the label or the bounding box based on the corresponding divided images. So, make sure the image is already labelled/annotated (YOLO format). 

![image](https://github.com/ismarapw/tile-image-yolo/assets/76652264/0618187d-4a50-4f08-9369-a3bb3830b88c)

## Run Script
1. Clone the repo and make sure you have opencv and python installed.
2. Place your image and the label in data folder
3. Open generate_data.py and change the IMAGE_DIR and LABEL_DIR according to yours
4. Specify the grid size (3x3, 5x5, 7x7, ...)
5. Run generate_data.py script (python3 generate_data.py)
6. The image and the label will be generated and saved in generated folder

## Current Limitation
1. Tiling image and feed them to the network will produce model that can only recognize the divided image, not a whole image. So, in the future i want to make a script to automate tile and merge process during inference.
2. Only single image can be feeded to the script. Soon, maybe the script can accept multiple images.
