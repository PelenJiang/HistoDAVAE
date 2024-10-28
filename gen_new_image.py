import argparse
import numpy as np
import os
from utils import *


def crop_to_rect(img, mask, px_margin=128):
    r"""Crops mask to rectangle"""

    x, y = np.where(mask)
    x_min, x_max = np.min(x), np.max(x)  
    y_min, y_max = np.min(y), np.max(y)  

    x_min = max(0, x_min - px_margin)
    x_max = min(mask.shape[0], x_max + px_margin)
    y_min = max(0, y_min - px_margin)
    y_max = min(mask.shape[1], y_max + px_margin)


    cropped_mask = mask[x_min:x_max, y_min:y_max]  
    cropped_img = img[x_min:x_max, y_min:y_max]    

    return x_min,y_min,cropped_img,cropped_mask

if __name__ == '__main__':


    Processed_sample_dir = 'data/Processed_HER2/'
    sample_name = 'H1'
    sample_path = os.path.join(Processed_sample_dir, sample_name)

    image = load_image(f'{sample_path}/{sample_name}_raw.jpg')
    mask = load_mask(f'{sample_path}/{sample_name}_first_mask.png')
    print(mask.shape)

    # x_min and y_min represent row_min and column_min
    x_min, y_min, image, mask = crop_to_rect(image,mask)  

    cropped_loc_file = open(f'{sample_path}/{sample_name}_cropped_locs.txt',mode='a+')
    loc_path = f'{sample_path}/{sample_name}_locs.txt'
    lines = read_lines(loc_path)


    for line in lines:
        x_coord = float(line.split(',')[0])-y_min
        y_coord = float(line.split(',')[1])-x_min
        cropped_loc_file.write(str(x_coord)+','+str(y_coord)+'\n')
    
    save_image(image,f'{sample_path}/{sample_name}_cropped_image.jpg')

    save_image(mask,f'{sample_path}/{sample_name}_cropped_mask.png')