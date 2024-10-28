import argparse
import os
import numpy as np
from einops import reduce

from utils import load_image, save_image, load_mask

def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img


def adjust_margins(img, pad, pad_value=None):  

    extent = np.stack([[0, 0], img.shape[:2]]).T       
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(
            img, extent, mode=mode, constant_values=pad_value)
    return img


def reduce_mask(mask, factor):
    mask = reduce(
            mask.astype(np.float32),
            '(h0 h1) (w0 w1) -> h0 w0', 'mean',
            h1=factor, w1=factor) > 0.5
    return mask



if __name__ == '__main__':

    Processed_sample_dir = 'data/Processed_HER2/'
    sample_name = 'H1'
    sample_path = os.path.join(Processed_sample_dir, sample_name)
    
    pad = 32
    img = load_image(f'{sample_path}/{sample_name}_cropped_image.jpg')
    img = adjust_margins(img, pad=pad, pad_value=255)   
    # save final histology image
    save_image(img, f'{sample_path}/{sample_name}_final_he.jpg')

    # load tissue mask
    mask = load_mask(f'{sample_path}/{sample_name}_cropped_mask.png')
    mask = adjust_margins(mask, pad=pad, pad_value=mask.min())
    # save tissue mask
    save_image(mask, f'{sample_path}/{sample_name}_final_mask.png')
    mask = reduce_mask(mask, factor=32)
    save_image(mask, f'{sample_path}/{sample_name}_mask-small.png')
