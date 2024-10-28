import itertools
from PIL import Image
import pickle
import os

import torch
import scanpy as sc
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from einops import rearrange
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_string(string, filename):
    return write_lines([string], filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)



def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)


def load_yaml(filename, verbose=False):
    with open(filename, 'r') as file:
        content = yaml.safe_load(file)
    if verbose:
        print(f'YAML loaded from {filename}')
    return content


def save_yaml(filename, content):
    with open(filename, 'w') as file:
        yaml.dump(content, file)
    print(file)

def plot_super(cmap_name, bool_mask, x, outfile, underground=None, truncate=None):

    x = x.copy()

    if truncate is not None:
        x -= np.nanmean(x)
        x /= np.nanstd(x) + 1e-12
        x = np.clip(x, truncate[0], truncate[1])

    x -= np.nanmin(x)
    x /= np.nanmax(x) + 1e-12

    
    cmap = plt.get_cmap(cmap_name)
    if underground is not None:
        under = underground.mean(-1, keepdims=True)
        under -= under.min()
        under /= under.max() + 1e-12

    img = cmap(x)[..., :3]
    if underground is not None:
        img = img * 0.5 + under * 0.5
    img[~bool_mask] = 1.0
    img = (img * 255).astype(np.uint8)
    save_image(img, outfile)

def get_image_patches(filename,patch_size):
    '''
      Splitting whole slide image into patches
    '''
    print('Splitting image into patches...')
    # t0 = time()
    wsi = load_image(filename)
    print('Image shape:', wsi.shape)
    patches = rearrange(
            wsi, '(h1 p_h) (w1 p_w) k -> h1 w1 p_h p_w k',
            p_h=patch_size, p_w=patch_size)    
    
    return patches

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def get_disk_mask(radius, boundary_width=None): 
    radius_ceil = np.ceil(radius).astype(int)     
    locs = np.meshgrid(        
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    # print("locs是什么",locs)
    locs = np.stack(locs, -1)     
    distsq = (locs**2).sum(-1)    
    isin = distsq <= radius**2    
    if boundary_width is not None:  
        isin *= distsq >= (radius-boundary_width)**2

    return locs, isin       

def get_patches_flat(img, locs, mask,mask_locs):    
    shape = np.array(mask.shape)    
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    x_loc_list = []
    for s in locs:
        patch = img[         
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        patch_loc = mask_locs + s            
        if mask.all():   
            x = patch
        else:
            x = patch[mask]
            loc = patch_loc[mask]
        x_list.append(x)
        x_loc_list.append(loc)
    x_list = np.stack(x_list)
    x_loc_list = np.stack(x_loc_list)

    return x_list, x_loc_list    

class ST_WSI_Dataset(Dataset):

    def __init__(self, img, locs, loc_range):
        super().__init__()
        self.x = img
        self.raw_locs = locs
        self.locs = locs/img.shape[:2]*loc_range

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = self.x[idx]/255.0
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        return img, self.locs[idx]

class ST_Spot_Dataset(Dataset):

    def __init__(self, img, locs, cnts, size_factors, radius,loc_range):
        super().__init__()
        mask_locs, mask = get_disk_mask(radius) 
        p_img,p_loc = get_patches_flat(img, locs, mask, mask_locs) 
        self.x = p_img    
        self.y = cnts  
        self.spot_locs=locs
        self.raw_locs = p_loc
        self.locs = p_loc/img.shape[:2]*loc_range
        self.size_factors = size_factors
        self.radius = radius
        self.num_one_spot = self.x.shape[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]/255.0
        img = torch.from_numpy(img)
        img = img.permute(0, 3, 1, 2)
        return img, self.locs[idx], self.y[idx],self.size_factors[idx]
    
