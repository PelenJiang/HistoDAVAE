import os
import random
import argparse
import torch
import numpy as np
from sklearn.cluster import KMeans
import scanpy as sc
import time
from torch.utils.data import random_split

from HistoDAVAE import HistoDAVAE
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='HistoDAVAE: dependency-aware variational autoencoder for histology',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_pth', default='data/Processed_HER2/H1/')
    parser.add_argument('--result_dir', default='Results/HER2/H1/')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--num_select_genes', default=314, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--maxiter', default=1000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=40, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[256, 128], type=int)
    parser.add_argument('--GP_dim', default=2, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=18, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[128, 256], type=int)
    parser.add_argument('--dynamicVAE', default=False, type=bool, 
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta', default=1, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=0.5, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=5, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=12, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='Trained_model.pt')  
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--predict_only', action='store_true', help='Prediction Only Mode')
    parser.add_argument('--test_data', default='spots', choices=['spots', 'all'], type=str)
    parser.add_argument('--cmap_name', default='gist_ncar', type=str, help='Choose colormap name in matplotlib')
    args = parser.parse_args()
    return args

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def prepare_data(args):
    patch_size = 32 

    if not os.path.isfile(f'{args.result_dir}Spot_set.pkl'):
        img_path = f'{args.data_pth}H1_final_he.jpg'
        loc_path = f'{args.data_pth}H1_cropped_locs.txt'
        radius_path = f'{args.data_pth}H1_radius.txt'
        adata  = sc.read(f'{args.data_pth}H1_whole_genes.h5ad')
        gene_list = list(np.load(f'{args.data_pth}H1_gene_svg314.npy',allow_pickle=True))

        adata = adata[:,gene_list].copy()

        image_patch = get_image_patches(img_path,patch_size) 
        loc_in_line = read_lines(loc_path)
        locs = []
        for loc in loc_in_line:
            loc = [float(loc.split(',')[1]),float(loc.split(',')[0])]
            locs.append(loc)
        coordinates = np.stack(locs)
        coordinates /= patch_size
        coordinates = coordinates.round().astype(int)  
        
        radius = float(read_string(radius_path))
        radius = radius/patch_size

        adata = normalize(adata,
                        filter_min_counts=False,
                        size_factors=True,
                        normalize_input=False,
                        logtrans_input=False)

        Spot_set = ST_Spot_Dataset(img=image_patch, locs=coordinates, cnts=adata.raw.X, size_factors=adata.obs.size_factors, radius=radius, loc_range = args.loc_range)

        loc_wsi_patches = np.empty((image_patch.shape[0],image_patch.shape[1],2))
        for x in range(image_patch.shape[0]):
            for y in range(image_patch.shape[1]):
                loc_wsi_patches[x, y] = [x, y]
        loc_wsi_patches = loc_wsi_patches.astype(int)
        mask = load_image(f'{args.data_pth}H1_mask-small.png')
        image_patch = image_patch[mask]
        loc_wsi_patches = loc_wsi_patches[mask]
        WSI_set = ST_WSI_Dataset(img=image_patch, locs=loc_wsi_patches,loc_range = args.loc_range) 
        save_pickle(Spot_set,f'{args.result_dir}Spot_set.pkl')
        save_pickle(WSI_set,f'{args.result_dir}WSI_set.pkl')
    else:
        Spot_set = load_pickle(f'{args.result_dir}Spot_set.pkl')
        print("Spot_set has been loaded!")

    # Split spot dataset if train size < 1 
    if args.train_size == 1:
        spot_dataset = {"train": Spot_set}
        
    elif args.train_size < 1:
        print("Spliting dataset according to the ratio:", args.train_size)
        if not os.path.isfile(f'{args.result_dir}Val_set.pkl'):
            Train_set, Val_set = random_split(dataset=Spot_set, lengths=[args.train_size, 1-args.train_size],
                                                        generator= torch.Generator().manual_seed(args.seed))
            save_pickle(Train_set,f'{args.result_dir}Train_set.pkl')
            save_pickle(Val_set,f'{args.result_dir}Val_set.pkl')
        else:
            Train_set = load_pickle(f'{args.result_dir}Train_set.pkl')
            Val_set = load_pickle(f'{args.result_dir}Val_set.pkl')
            print("Train and Val set has been loaded!")
        spot_dataset = {"train": Train_set, "val": Val_set}
    else:
        raise ValueError(f"Input train_size is invalid! ")

    return spot_dataset

def predict(model,args):
    mask = load_image(f'{args.data_pth}H1_mask-small.png')
    gene_list = list(np.load(f'{args.data_pth}H1_gene_svg314.npy',allow_pickle=True))
    if args.test_data == 'spots':
        Spot_set = load_pickle(f'{args.result_dir}Spot_set.pkl')
        predict_genes = model.batching_predict_train_genes(Spot_set, args.batch_size)
        name_list = gene_list
        predict_genes = rearrange(predict_genes, 'b n w -> (b n) w')
        loc_for_pre = rearrange(Spot_set.raw_locs, 'b n w -> (b n) w')
        np.save(f'{args.result_dir}/loc_finally_train.npy',loc_for_pre)

        for i in range(len(name_list)):
            bool_mask = (mask.copy() > 0)
            mask_for_pre = np.zeros(mask.shape)
            gene_name = name_list[i]
            mkdirs(f'{args.result_dir}super-spot-patches-{args.cmap_name}')
            mkdirs(f'{args.result_dir}super-spot-patches-Value')
            out_path =  f'{args.result_dir}super-spot-patches-{args.cmap_name}/{gene_name}.png'
            out_path_tsv =  f'{args.result_dir}super-spot-patches-Value/{gene_name}.csv'
            for patch_id in range(len(loc_for_pre)):
                # print(f"patch_id: {patch_id}")
                loc_p = loc_for_pre[patch_id]
                mask_for_pre[loc_p[0]][loc_p[1]] =  predict_genes[patch_id][i]
                
            plot_super(args.cmap_name,bool_mask,mask_for_pre,out_path)
            # plot_super_with_label(bool_mask,mask_for_pre,out_path)
            np.savetxt(out_path_tsv, mask_for_pre, delimiter=',')

            del mask_for_pre

    elif args.test_data == 'all':
        WSI_set = load_pickle(f'{args.result_dir}WSI_set.pkl')
        predict_genes = model.batching_predict_WSI_genes(WSI_set, args.batch_size)
        name_list = gene_list
        loc_wsi_patches = WSI_set.raw_locs
        for i in range(len(name_list)):
            bool_mask = (mask.copy() > 0)
            mask_for_pre = np.zeros(mask.shape)
            gene_name = name_list[i]
            mkdirs(f'{args.result_dir}super-all-patches-{args.cmap_name}')
            mkdirs(f'{args.result_dir}super-all-patches-Value')
            out_path =  f'{args.result_dir}super-all-patches-{args.cmap_name}/{gene_name}.png'
            out_path_tsv =  f'{args.result_dir}super-all-patches-Value/{gene_name}.csv'

            for patch_id in range(len(loc_wsi_patches)):
                loc_p = loc_wsi_patches[patch_id]
                mask_for_pre[loc_p[0]][loc_p[1]] =  predict_genes[patch_id][i]

            plot_super(args.cmap_name,bool_mask,mask_for_pre,out_path)
            np.savetxt(out_path_tsv, mask_for_pre, delimiter=',')

            del mask_for_pre
    


def main():
    # Set the hyper parameters and random seed
    args = get_args()
    set_seed(seed=args.seed)

    mkdirs(args.result_dir)

    start_time = time.time()

    # Prepare data for model training and prediction
    spot_dataset = prepare_data(args)

    # Generate inducing points for SVGP
    if args.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range

    else:
        if args.train_size == 1:
            loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(spot_dataset["train"].locs)
        else:
            loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(spot_dataset["train"].dataset.locs)
        np.savetxt(f'{args.result_dir}location_centroids.txt', loc_kmeans.cluster_centers_, delimiter=",")
        np.savetxt(f'{args.result_dir}location_kmeans_labels.txt', loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    # Build model
    if args.train_size == 1:
        N_train_patches = len(spot_dataset["train"]) * spot_dataset["train"].num_one_spot
    else:
        N_train_patches = len(spot_dataset["train"]) * spot_dataset["train"].dataset.num_one_spot
    model = HistoDAVAE(GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, Predict_dim = args.num_select_genes, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
    fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
    fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=N_train_patches, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE, init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta, dtype=torch.float64, device=args.device)

    # Train and test model
    if args.predict_only:
        model.load_model(args.model_file)
        predict(model,args)
    else:
        train_start_time = time.time()
        model.train_model(st_dataset=spot_dataset, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
        print('Training time: %d seconds.' % int(time.time() - train_start_time))
        predict(model,args)

    end_time = time.time()  # end time
    elapsed_time = int(end_time - start_time)
    print(f"Total running time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()




