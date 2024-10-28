import argparse
from time import time

import numpy as np
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from scipy.ndimage import label
from PIL import Image
from typing import Sequence, Dict, List, NamedTuple, Optional, Tuple
import itertools as it
from utils import load_image


from scipy.ndimage.morphology import binary_fill_holes

def rescale(
    image, scaling_factor, resample):

    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize(
        [round(x * scaling_factor) for x in image_pil.size], resample=resample,
    )
    return np.array(image_pil)


def resize(image, target_shape, resample):

    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize(target_shape[::-1], resample=resample)
    return np.array(image_pil)

def find_min_bbox(mask, rotate):
    
    contour, _ = cv.findContours(
        mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if rotate:
        return cv.minAreaRect(np.concatenate(contour))
    x, y, w, h = cv.boundingRect(np.concatenate(contour))
    return ((x + w // 2, y + h // 2), (w, h), 0.0)


def crop_to_rect(img, rect, interpolation_method, margin):

    width, height = rect[1]
    px_margin = margin * max(width, height)
    width = int(np.round(width + 2 * px_margin))
    height = int(np.round(height + 2 * px_margin))
    rect = (rect[0], (width, height), rect[2])
    box_src = cv.boxPoints(rect)
    box_dst = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype=np.float32,
    )
    transform = cv.getPerspectiveTransform(box_src, box_dst)
    return cv.warpPerspective(
        img,
        transform,
        (width, height),
        flags=interpolation_method,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=np.median(
            np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), 0
        ),
    )

def remove_fg_elements(mask, size_threshold):

    labels, _ = label(mask)
    labels_unique, label_counts = np.unique(labels, return_counts=True)
    small_labels = labels_unique[
        label_counts < size_threshold ** 2 * np.prod(mask.shape)
    ]
    mask[np.isin(labels, small_labels)] = False
    return mask

def compute_tissue_mask(image,convergence_threshold = 0.0001,size_threshold = 0.01,initial_mask = None,
):

    original_shape = image.shape[:2]
    scale_factor = 1000 / max(original_shape) 

    image = rescale(image, scale_factor, resample=Image.NEAREST)  

    if initial_mask is None:
        initial_mask = (
            cv.blur(cv.Canny(cv.blur(image, (5, 5)), 100, 200), (20, 20)) > 0  
        )
        initial_mask = binary_fill_holes(initial_mask)  
        initial_mask = remove_fg_elements(initial_mask, 0.1) 
        mask = np.where(initial_mask, cv.GC_PR_FGD, cv.GC_PR_BGD)
        mask = mask.astype(np.uint8)
    else:
        mask = initial_mask
        mask = rescale(mask, scale_factor, resample=Image.NEAREST)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = bgd_model.copy()

    print("Computing tissue mask:")

    for i in it.count(1):
        old_mask = mask.copy()
        try:
            cv.grabCut(  
                image,
                mask,
                None,
                bgd_model,
                fgd_model,
                1,
                cv.GC_INIT_WITH_MASK,
            )
        except cv.error as cv_err:
            print(f"Failed to mask tissue\n{str(cv_err).strip()}")
            mask = np.full_like(mask, cv.GC_PR_FGD)
            break
        prop_changed = (mask != old_mask).sum() / np.prod(mask.shape)
        print("  Iteration {:03d} Δ = {:.3f}".format(i, 100 * prop_changed))
        if prop_changed < convergence_threshold:
            break

    mask = np.isin(mask, [cv.GC_FGD, cv.GC_PR_FGD])  
    mask = cleanup_mask(mask, size_threshold)

    mask = resize(mask, target_shape=original_shape, resample=Image.NEAREST)

    return mask

def cleanup_mask(mask, size_threshold):
    
    mask = ~remove_fg_elements(~mask, size_threshold)
    mask = remove_fg_elements(mask, size_threshold)
    return mask


if __name__ == '__main__':


    Processed_sample_dir = 'data/Processed_HER2/'
    sample_name = 'H1'

    img_path = f'{Processed_sample_dir}/{sample_name}/{sample_name}_raw.jpg'
    image = load_image(img_path)
    mask=compute_tissue_mask(image)
    mask=Image.fromarray(mask)
    mask.save(f'{Processed_sample_dir}/{sample_name}/{sample_name}_first_mask.png')