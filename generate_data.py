import cv2
from utils import *

if __name__ == "__main__":
    IMG_DIR = 'data/images/DSC06145.JPG'
    LABEL_DIR = 'data/labels/DSC06145.txt'

    img = cv2.imread(IMG_DIR)

    bbox_list = read_bbox(LABEL_DIR)
    bbox_denormalized = denormalize_bbox(bbox_list, img)

    grid_size = (5,5)
    img_width = img.shape[1]
    img_height = img.shape[0]
    chunks_width = get_chunks_width(img_width, grid_size[1])
    chunks_height = get_chunks_height(img_height, grid_size[0])
    tiled_image = tile_images(img, chunks_width, chunks_height, bbox_denormalized)

    save_tile_image(tiled_image, target_dir='generated')