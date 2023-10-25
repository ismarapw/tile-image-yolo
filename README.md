# Tile Image for YOLO Format
Sometimes small object in large resolution image need to be detected. If we try to labels the small object in a high resolution image and feed the data to the network with certain size (416x416 for example), then it will lose the detail of the small object because it is going to be compressed. As a result, the network cannot detect the object properly. 

We can try to set the network input size based on the original image resolution, but most of the network architecture for object detection has constraint input size. The larger input size used in the network the larger memory in GPU will be used for training process. This is not very efficient for certain condition especially for those who has GPU memory limitation. 

This repo try to solve the problem by using image pre-processing technique called tile images. This such type of image processing can divide an image to several chunks images based on the grid size (3x3, 5x5, etc). The script also divide the label or the bounding box based on the corresponding divided images. So, make sure the image is already labelled/annotated (YOLO format). 

# Run Script
1. Clone the repo and make sure you have opencv and python installed.
2. Place your image and the label in data folder
3. Open generate_data.py and change the IMAGE_DIR and LABEL_DIR according to yours
4. Run generate_data.py script (python3 generate_data.py)
5. The image and the label will be generated and saved in generated folder

