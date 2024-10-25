
import os
from time import time
import argparse
import numpy as np
import cv2 as cv
import torch
from typing import Dict, List, NamedTuple, Optional, Tuple
from torch import nn
import skimage
from scipy.ndimage import uniform_filter
from einops import rearrange, reduce, repeat
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from utils import (
        load_tsv, load_image, save_image,read_string,
        read_lines)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



def impute_missing(x, mask, radius=3, method='ns'):

    method_dict = {
            'telea': cv.INPAINT_TELEA,
            'ns': cv.INPAINT_NS}
    method = method_dict[method]

    x = x.copy()
    if x.dtype == np.float64:
        x = x.astype(np.float32)

    x[mask] = 0
    mask = mask.astype(np.uint8)

    expand_dim = np.ndim(x) == 2
    if expand_dim:
        x = x[..., np.newaxis]
    channels = [x[..., i] for i in range(x.shape[-1])]
    y = [cv.inpaint(c, mask, radius, method) for c in channels]
    y = np.stack(y, -1)
    if expand_dim:
        y = y[..., 0]

    return y


def smoothen(
        x, size, kernel='gaussian', backend='cv', mode='mean',
        impute_missing_values=True, device='cuda'):

    if x.ndim == 3:
        expand_dim = False
    elif x.ndim == 2:
        expand_dim = True
        x = x[..., np.newaxis]
    else:
        raise ValueError('ndim must be 2 or 3')

    mask = np.isfinite(x).all(-1)
    if (~mask).any() and impute_missing_values:
        x = impute_missing(x, ~mask)

    if kernel == 'gaussian':
        sigma = size / 4  # approximate std of uniform filter 1/sqrt(12)
        truncate = 4.0
        winsize = np.ceil(sigma * truncate).astype(int) * 2 + 1
        if backend == 'cv':
            print(f'gaussian filter: winsize={winsize}, sigma={sigma}')
            y = cv.GaussianBlur(
                    x, (winsize, winsize), sigmaX=sigma, sigmaY=sigma,
                    borderType=cv.BORDER_REFLECT)
        elif backend == 'skimage':
            y = skimage.filters.gaussian(
                    x, sigma=sigma, truncate=truncate,
                    preserve_range=True, channel_axis=-1)
        else:
            raise ValueError('backend must be cv or skimage')
    elif kernel == 'uniform':
        if backend == 'cv':
            kernel = np.ones((size, size), np.float32) / size**2
            y = cv.filter2D(
                    x, ddepth=-1, kernel=kernel,
                    borderType=cv.BORDER_REFLECT)
            if y.ndim == 2:
                y = y[..., np.newaxis]
        elif backend == 'torch':
            assert isinstance(size, int)
            padding = size // 2
            size = size + 1

            pool_dict = {
                    'mean': nn.AvgPool2d(
                        kernel_size=size, stride=1, padding=0),
                    'max': nn.MaxPool2d(
                        kernel_size=size, stride=1, padding=0)}
            pool = pool_dict[mode]

            mod = nn.Sequential(
                    nn.ReflectionPad2d(padding),
                    pool)
            y = mod(torch.tensor(x, device=device).permute(2, 0, 1))
            y = y.permute(1, 2, 0)
            y = y.cpu().detach().numpy()
        else:
            raise ValueError('backend must be cv or torch')
    else:
        raise ValueError('kernel must be gaussian or uniform')

    if not mask.all():
        y[~mask] = np.nan

    if expand_dim and y.ndim == 3:
        y = y[..., 0]

    return y


def upscale(x, target_shape):
    mask = np.isfinite(x).all(tuple(range(2, x.ndim)))
    x = impute_missing(x, ~mask, radius=3)
    # TODO: Consider using pytorch with cuda to speed up
    # order: 0 == nearest neighbor, 1 == bilinear, 3 == bicubic
    dtype = x.dtype
    x = skimage.transform.resize(
            x, target_shape, order=3, preserve_range=True)
    x = x.astype(dtype)
    if not mask.all():
        mask = skimage.transform.resize(
                mask.astype(float), target_shape, order=3,
                preserve_range=True)
        mask = mask > 0.5
        x[~mask] = np.nan
    return x


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


def get_disk_mask(radius, boundary_width=None):
    #下面的代码什么意思：
    radius_ceil = np.ceil(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    # print("locs是什么",locs)
    locs = np.stack(locs, -1)
    # print("stack后locs是什么",locs.shape,locs)

    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    # print("isin是什么",isin.shape,isin)
    #返回一个15x15的矩阵（半径为7的外接矩形），里面有True和False，True表示该位置是属于spot的位置，False表示不是
    return isin


def shrink_mask(x, size):
    size = size * 2 + 1
    x = uniform_filter(x.astype(float), size=size)
    x = np.isclose(x, 1)
    return x

def get_data(prefix):
    img = load_image(f'{prefix}he.jpg')
    return img


def get_image_patches(filename):
    '''
      Splitting whole slide image into patches
    '''
    print('Splitting image into patches...')
    # t0 = time()
    wsi = load_image(filename)
    print('Image shape:', wsi.shape)
    patches = rearrange(
            wsi, '(h1 p_h) (w1 p_w) k -> h1 w1 p_h p_w k',
            p_h=32, p_w=32)    
    
    return patches


def get_gene_counts(prefix, reorder_genes=True):
    cnts = load_tsv(f'{prefix}cnts.tsv')
    if reorder_genes:
        order = cnts.var().to_numpy().argsort()[::-1]
        cnts = cnts.iloc[:, order]
    return cnts

def get_locs(prefix, target_shape=None):

    locs = load_tsv(f'{prefix}locs.tsv')

    # change xy coordinates to ij coordinates
    locs = np.stack([locs['y'], locs['x']], -1)

    # match coordinates of embeddings and spot locations
    if target_shape is not None:
        wsi = load_image(f'{prefix}he.jpg')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs = locs.astype(float)
        locs /= rescale_factor

    # find the nearest pixel
    locs = locs.round().astype(int)

    return locs

def get_train_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    patches = get_image(prefix)
    locs = get_locs(prefix, target_shape=patches.shape[:2])
    # embs = add_coords(embs)
    return patches, cnts, locs

    
class ST_WSI_Dataset(Dataset):

    def __init__(self, img, locs, loc_range,mask):
        super().__init__()
        # print("img.shape:",img.shape)

        self.x = img[mask]
        self.locs = locs/img.shape[:2]*loc_range
        # print("locs:", self.locs)

    def __len__(self):
        return self.x.shape[0]


    def __getitem__(self, idx):
        img = self.x[idx]/255.0
        img = torch.from_numpy(img)
        # print("img.shape:",img.shape)
        img = img.permute(2, 0, 1)
        # if self.norm:
        #     img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        return img, self.locs[idx]

class ST2Dataset(Dataset):

    def __init__(self, img, locs, cnts, size_factors, radius,loc_range):
        super().__init__()
        mask_locs, mask = get_disk_mask(radius) # #用于生成一个圆形的掩码，radius圆的半径，返回所有像素点的坐标mask_locs，对应的掩码数组mask
        # print("mask的shape",mask.shape)
        # print("mask是什么",mask)
        p_img,p_loc = get_patches_flat(img, locs, mask, mask_locs) # 从输入图像中提取符合条件的图像patch, 返回图像patch的数组p_img和对应的位置p_loc
        self.x = p_img    #图像
        self.y = cnts     #表达数据
        # print("locs之前",p_loc)
        self.locs = p_loc/img.shape[:2]*loc_range
        # print("locs之后",self.locs)
        self.size_factors = size_factors
        self.radius = radius

    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        img = self.x[idx]/255.0
        img = torch.from_numpy(img)
        img = img.permute(0, 3, 1, 2)
        # if self.norm:
        #     img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        return img, self.locs[idx], self.y[idx],self.size_factors[idx]
    
class STDataset(Dataset):

    def __init__(self, img, locs, cnts, size_factors, radius, normalize=False):
        super().__init__()
        # x, x_loc =  get_patches(img, locs, radius)
        self.norm = normalize
        self.x = img
        self.locs = locs
        self.y = cnts
        self.size_factors = size_factors
        self.radius = radius

    def __len__(self):
        return len(self.x)
    
    def transform(self, image):
        # image = Image.fromarray(image)
        # # Random flipping and rotations
        # if random.random() > 0.5:
        #     image = TF.hflip(image)
        # if random.random() > 0.5:
        #     image = TF.vflip(image)
        # angle = random.choice([180, 90, 0, -90])
        # image = TF.rotate(image, angle)
        return np.asarray(image)

    def __getitem__(self, idx):
        img = self.x[idx]/255.0
        img = torch.from_numpy(img)
        img = img.permute(0, 3, 1, 2)
        # if self.norm:
        #     img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        return img, self.locs[idx], self.y[idx],self.size_factors[idx]

# def get_patches(img, locs, radius):
#     # print('locs:', locs)
#     # patch_size = int(radius/2)
#     patch_size = int(32)
#     # print(patch_size)
#     patch_list = []
#     patch_locs = []
#     for s in locs:
#         # print("s:", s)
#         # print('patch_11_x:',s[0]-patch_size,s[0])
#         # print('patch_11_y:',s[1]-patch_size,s[1])
#         s_patch = []
#         s_loc = []
#         patch_11 = img[
#                 s[0]-patch_size:s[0],
#                 s[1]-patch_size:s[1]]
#         # print("patch_11:", patch_11.shape)
#         patch_11_loc = [s[0]-patch_size/2,s[1]-patch_size/2]
#         s_patch.append(patch_11)
#         s_loc.append(patch_11_loc)        
#         patch_12 = img[
#                 s[0]-patch_size:s[0],
#                 s[1]:s[1]+patch_size]
#         patch_12_loc = [s[0]-patch_size/2,s[1]+patch_size/2]
#         s_patch.append(patch_12)
#         s_loc.append(patch_12_loc)
#         patch_21 = img[
#                 s[0]:s[0]+patch_size,
#                 s[1]-patch_size:s[1]]
#         patch_21_loc = [s[0]+patch_size/2,s[1]-patch_size/2]
#         s_patch.append(patch_21)
#         s_loc.append(patch_21_loc)
#         patch_22 = img[
#                 s[0]:s[0]+patch_size,
#                 s[1]:s[1]+patch_size]
#         patch_22_loc = [s[0]+patch_size/2,s[1]+patch_size/2]
#         s_patch.append(patch_22)
#         s_loc.append(patch_22_loc)
#         s_patch = np.stack(s_patch)
#         s_loc = np.stack(s_loc)
#         patch_list.append(s_patch)
#         patch_locs.append(s_loc)
#         del s_patch, s_loc
#     patch_list = np.stack(patch_list).astype(np.float64)
#     patch_locs = np.stack(patch_locs).astype(np.float64)
    # print("patch_list:", patch_list.shape)
    # print("patch_list.type:", type(patch_locs))
    # print("patch_list.type:", type(patch_locs[0][0][0][0]))

    # return patch_list, patch_locs

def get_patches_flat(img, locs, mask,mask_locs):    # 从输入图像中提取符合条件的图像patch, 并返回这些patch对应的位置信息
    shape = np.array(mask.shape)    
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    x_loc_list = []
    for s in locs:
        # print('s:', s)
        patch = img[                         # 从img中切出的patch
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        # print('patch:', patch.shape)
        patch_loc = mask_locs + s            # s： patch的位置信息，与mask_locs得到绝对位置
        # print("patch_loc:", patch_loc, patch_loc.shape)
        if mask.all():   #如果mask的所有元素都为True,则直接将整个patch添加到x_list中
            x = patch
        else:
            x = patch[mask]
            loc = patch_loc[mask]
            # print("patch_loc:", loc, loc.shape)
        x_list.append(x)
        x_loc_list.append(loc)
    x_list = np.stack(x_list)
    x_loc_list = np.stack(x_loc_list)
    # print('x_list:', x_list.shape)
    # print('x_loc_list:', x_loc_list.shape)
    return x_list, x_loc_list    #返回图像patch的数组和对应的位置

def get_disk_mask(radius, boundary_width=None):  #用于生成一个圆形的掩码，radius圆的半径，boundary_width圆的边界宽度

    radius_ceil = np.ceil(radius).astype(int)     
    locs = np.meshgrid(        #生成网络坐标
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    # print("locs是什么",locs)
    locs = np.stack(locs, -1)     #堆叠坐标
    # print("stack后locs是什么",locs.shape,locs)

    distsq = (locs**2).sum(-1)    #计算距离平方
    isin = distsq <= radius**2    #生成掩码
    if boundary_width is not None:   #考虑边界宽度
        isin *= distsq >= (radius-boundary_width)**2

    return locs, isin       #返回所有像素点的坐标locs，对应的掩码数组isin

def get_gene_counts(count_path, reorder_genes=True):
    cnts = load_tsv(count_path)
    # print("cnts.shape:",cnts)
    if reorder_genes:
        order = cnts.var().to_numpy().argsort()[::-1]
        cnts = cnts.iloc[:, order]
        # print("cnts.shape:",cnts)

    return cnts

def get_ST_data(count_path, image_path, gene_list_path, transformation_path=None):
    gene_names = read_lines(gene_list_path)  # HVG基因
    counts = get_gene_counts(count_path)     # count数据，即所有的基因表达数据
    image = load_image(image_path)           # 图片
    counts = counts[gene_names]              # 筛选只剩HVG基因的count数据
    # print("counts.shape:",counts.shape,type(counts))
    if transformation_path is not None:
        transformation = np.loadtxt(transformation_path)
        transformation = transformation.reshape(3, 3)
    else:
        transformation = None
    return counts, image, transformation

def get_img_loc_cnts(count_path, image_path, gene_list_path, spot_loc_path):
    gene_names = read_lines(gene_list_path)
    counts = get_gene_counts(count_path)
    image = get_image_patches(image_path)
    counts = counts[gene_names]
    # print("counts.shape:",counts.shape,type(counts))
    loc_in_line = read_lines(spot_loc_path)
    locs = []
    for loc in loc_in_line:
        loc = [float(loc.split(',')[1]),float(loc.split(',')[0])]
        locs.append(loc)
    locs = np.stack(locs)
    # print("locs.shape:",locs.shape,type(locs),locs)
    return counts, image, locs

def generate_mask(image_path, gene_list_path, transformation_path=None):
    gene_names = read_lines(gene_list_path)
    counts = get_gene_counts(count_path)
    image = load_image(image_path)
    counts = counts[gene_names]
    # print("counts.shape:",counts.shape,type(counts))
    if transformation_path is not None:
        transformation = np.loadtxt(transformation_path)
        transformation = transformation.reshape(3, 3) #原始数据是9维的向量，重构为3*3
    else:
        transformation = None
    return counts, image, transformation

def get_ST_coord(counts,image, transformation=None):
    
    coord_list = []
    for x in counts.index:
        x_split = x.split("x")
        coord_list.append([float(x_split[0]), float(x_split[1])])

    coordinates = np.array(coord_list)
    space_coord = coordinates

    if transformation is not None:
            coordinates = np.concatenate(
                [coordinates, np.ones((len(coordinates), 1))], axis=-1
            )
            coordinates = coordinates @ transformation  #矩阵乘法运算
            coordinates = coordinates[:, :2]            #得到实际图像中的坐标
    else:
            coordinates[:, 0] = (coordinates[:, 0] - 1) / 32 * image.shape[1]
            coordinates[:, 1] = (coordinates[:, 1] - 1) / 34 * image.shape[0]
    # radius = np.sqrt(np.product(image.shape[:2]) / 32 / 34) / 4
    radius = np.sqrt(np.prod(image.shape[:2]) / 32 / 34) / 4    #这是什么？
    radius_file = open('/home/nas2/path/19_CYQ/code/gene_expression_spaVAE_clean/spaVAE/processed_data/MOB/Rep2_radius.txt',mode='w+')
    radius_file.write(str(radius))
    # spots = [Spot(x=x, y=y, r=radius) for x, y in coordinates]
    spot_log_file = open('/home/nas2/path/19_CYQ/code/gene_expression_spaVAE_clean/spaVAE/processed_data/MOB/Rep2_loc.txt',mode='a+')

    for i in range(len(coordinates)):
        x, y = coordinates[i]         #将实际坐标写入文件中
        # print("x,y分别是多少：",x,y)
        s_x, s_y = space_coord[i]     #将spot坐标(即spot的x,y序号)写入文件中
        # print("s_x,s_y分别是多少：",s_x,s_y)
        spot_log_file.write(str(x)+','+str(y)+','+str(s_x)+','+str(s_y)+'\n')

    return coordinates,radius

    # plt.imshow(image)
    # ax = plt.gca()
    
    # for spot in  spots:
    #     print("spot.x,spot.y,spot.r分别是多少：",spot.x,spot.y,spot.r)
    #     circle = plt.Circle((spot.x,spot.y), spot.r, color='r', fill=False)
    #     ax.add_artist(circle)
        
    # plt.savefig('spot_on_image.jpg')



# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--prefix', type=str, default='Data/demo/')
#     parser.add_argument('--epochs', type=int, default=None)  # e.g. 400
#     parser.add_argument('--n-states', type=int, default=2)
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--n-jobs', type=int, default=1)
#     parser.add_argument('--load-saved', action='store_true')
#     args = parser.parse_args()
    return args

if __name__ == '__main__':
    # args = get_args()

    patch_size = 32
    # img_path = 'Processed_Data/Rep3_he.jpg'
    # count_path = 'Data/MOB_data/Rep3_MOB_count.tsv'
    # gene_list_path = 'Data/MOB_data/Rep3_gene-names.txt'
    # loc_path = 'Processed_Data/cropped_spot_loc_Rep3.txt'

    img_path = '/home/nas2/path/19_CYQ/code/gene_expression_spaVAE_clean/spaVAE/Data/MOB_data/HE_Rep2.jpg'
    count_path = '/home/nas2/path/19_CYQ/code/gene_expression_spaVAE_clean/spaVAE/Data/MOB_data/Rep2_MOB_count.tsv'
    gene_list_path = '/home/nas2/path/19_CYQ/code/gene_expression_spaVAE_clean/spaVAE/processed_data/MOB/Rep2_gene-names.txt'
    transformation_path = '/home/nas2/path/19_CYQ/code/gene_expression_spaVAE_clean/spaVAE/Data/MOB_data/transform2.txt'

    counts, image, transformation = get_ST_data(count_path, img_path,gene_list_path, transformation_path)   # 得到ST数据，包含仅含hvg的count数据，图片，transformation
    coordinates,radius = get_ST_coord(counts,image, transformation)    #coordinates：实际坐标和spot坐标(即spot的x,y序号)       
    # counts, image, coordinates = get_img_loc_cnts(count_path, img_path,gene_list_path, loc_path)
    # radius = 72.62328039120383/32
    # coordinates = read_lines(spot_loc_path)
    # coordinates /= patch_size
    # coordinates = coordinates.round().astype(int)
    # coordinates = np.round(coordinates).astype(int)
    # print("image.shape是多少",image.shape)
    # print("coordinates是多少",coordinates.shape,coordinates)
    # int(round(x))
    # patch_list, patch_locs = get_patches(image, coordinates, radius)
    # imgs = Segment_patches(image)
    # Train_set = ST2Dataset(image, coordinates, counts, radius)

    # dataloader = DataLoader(
    #         Train_set, batch_size=2,
    #         shuffle=True)
    # dataloader_iter = iter(dataloader)
    # data = next(dataloader_iter)
    # print(f'Data shape: {data[0].shape},{type(data[0])}')
    # print(len(Train_set.x),len(Train_set.y),len(Train_set.x_loc))
    # print(Train_set.x[0],Train_set.x_loc[0],Train_set.y.shape)

