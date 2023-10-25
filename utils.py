import cv2

def read_bbox(dir):
    bbox_list = []

    with open(dir, 'r') as file:
        for line in file:
            line = line.strip() 
            parts = line.split()
            if len(parts) == 5: 
                bbox = [float(part) for part in parts] 
                bbox_list.append(bbox)
    
    return bbox_list

def denormalize_bbox(bbox_list, image):
    bbox_new = []
    for bbox in bbox_list :
        cls = bbox[0]
        cx = int(bbox[1] * image.shape[1])
        cy = int(bbox[2] * image.shape[0])
        w = int(bbox[3] * image.shape[1])
        h = int(bbox[4] * image.shape[0])

        bbox_new.append([int(cls), cx,cy,w,h])

    return bbox_new


def get_chunks_width(img_width, grid_col):
    chunks_width = [img_width // grid_col] * grid_col

    remainder_pixel = img_width % grid_col
    if remainder_pixel > 0:
        chunks_width[-1] += remainder_pixel

    return chunks_width


def get_chunks_height(img_height, grid_row):
    chunks_height = [img_height // grid_row] * grid_row

    remainder_pixel = img_height % grid_row
    if remainder_pixel > 0:
        chunks_height[-1] += remainder_pixel

    return chunks_height


def bbox_is_in_chunk(bbox, x, chunk_w, y, chunk_h):
    return bbox[1] > x and bbox[1] < x+chunk_w and bbox[2] > y and bbox[2] < y+chunk_h


def transform_bbox(bbox, x, chunk_w, y, chunk_h):
    bbox_cx = bbox[1] - x
    bbox_cy = bbox[2] - y
    bbox_w = bbox[3]
    bbox_h = bbox[4]

    x1 = bbox_cx - (bbox_w // 2)
    y1 = bbox_cy - (bbox_h // 2)
    x2 = bbox_cx + (bbox_w // 2)
    y2 = bbox_cy + (bbox_h // 2)

    # Shift bbox if the the current bbox is out of chunk width or hight
    if (x1 <= 0):
        bbox_cx = bbox_cx + abs(x1)
    if (x2 >= chunk_w):
        bbox_cx = bbox_cx - abs(chunk_w  - x2)
    if (y1 <= 0):
        bbox_cy = bbox_cy + abs(y1)
    if (y2 >= chunk_h):
        bbox_cy = bbox_cy - abs(chunk_h - y2)

    # normalize based on chunk w and chunk h
    bbox_cx = bbox_cx / chunk_w
    bbox_cy = bbox_cy / chunk_h
    bbox_w = bbox_w / chunk_w
    bbox_h = bbox_h / chunk_h

    return [bbox[0], bbox_cx, bbox_cy, bbox_w, bbox_h]


def tile_images(img, chunks_width, chunks_height, bbox_list):
    tiled_images = []

    y = 0

    for ch in chunks_height:
        x = 0
        for cw in chunks_width:
            chunk = img[y: y+ch, x:x+cw]
            bbox_in_chunk = []

            for bb in bbox_list:
                if bbox_is_in_chunk(bb, x, cw, y, ch):
                    transformed_bbox = transform_bbox(bb, x, cw, y, ch)
                    bbox_in_chunk.append(transformed_bbox)

            tiled_images.append([chunk, bbox_in_chunk])

            x += cw

        y += ch
    
    return tiled_images


def save_tile_image(tiled_img, target_dir):
    for i, chunk in enumerate(tiled_img):
        chunk_img = chunk[0]
        chunk_bbox = chunk[1]

        f = open(f"{target_dir}/chunk {i + 1}.txt","w+")

        for bb in chunk_bbox :
            f.write(f"{' '.join(str(item) for item in bb)}\r\n")

        f.close()
        
        cv2.imwrite(f'{target_dir}/chunk {i + 1}.jpg', chunk_img)