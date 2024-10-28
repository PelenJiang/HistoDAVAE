import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, random_split
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import numpy as np
import pandas as pd
from SVGP import SVGP
from I_PID import PIDControl
from VAE_utils import *
from collections import deque


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class HistoDAVAE(nn.Module):
    def __init__(self, GP_dim, Normal_dim, encoder_layers, decoder_layers, Predict_dim, encoder_dropout, decoder_dropout, 
                    fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, 
                    KL_loss, dynamicVAE, init_beta, min_beta, max_beta, dtype, device):
        super(HistoDAVAE, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        # self.input_dim = 3
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss          # expected KL loss value
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta           # beta controls the weight of reconstruction loss
        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.Predict_dim = Predict_dim  # dimension of predicted gene expression
        self.device = device
        self.encoder = ImageEncoder(hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = buildNetwork([GP_dim+Normal_dim]+decoder_layers, activation="elu", dropout=decoder_dropout)
        self.img_decoder = ImageDecoder(latent_dims=[GP_dim+Normal_dim])

        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], Predict_dim), MeanAct())
        else:
            self.dec_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, Predict_dim), MeanAct())
        self.dec_disp = nn.Parameter(torch.randn(self.Predict_dim), requires_grad=True)       # trainable dispersion parameter for NB loss

        self.NB_loss = NBLoss().to(self.device)
        self.MSE_loss = nn.MSELoss(reduction='sum')
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def forward(self, x,  y, raw_y, size_factors, num_samples=1):
        """
        Forward pass.

        Parameters:
        -----------
        x: mini-batch of positions.
        y: mini-batch of image patches.
        raw_y: mini-batch of raw counts.
        size_factor: mini-batch of size factors.
        num_samples: number of samplings of the posterior distribution of latent embedding.

        raw_y and size_factor are used for NB likelihood.
        """ 

        self.train()
        # change batch size to batch_size*num_patches
        x_b,x_num_p,loc = x.shape
        x = x.reshape(x_b*x_num_p,loc)
        y_b,y_num_p,y_c,h,w = y.shape
        y = y.reshape(y_b*y_num_p,y_c,h,w)


        qnet_mu, qnet_var = self.encoder(y)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l])
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l],
                                                                    noise=gp_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l)

            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            gp_p_m.append(gp_p_m_l)
            gp_p_v.append(gp_p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)

        inside_elbo = inside_elbo_recon - (y_b*y_num_p / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # cross entropy term
        gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var)
        gp_ce_term = torch.sum(gp_ce_term)

        # KL term of GP prior
        gp_KL_term = gp_ce_term - inside_elbo

        # KL term of Gaussian prior
        gaussian_prior_dist = Normal(torch.zeros_like(gaussian_mu), torch.ones_like(gaussian_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        gaussian_KL_term = kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))
        latent_samples = []
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        img_recon_loss = 0
        for f in latent_samples:
            recon_img_samples = self.img_decoder(f)
            img_recon_loss += self.MSE_loss(recon_img_samples, y )

        img_recon_loss = img_recon_loss / num_samples

        recon_loss = 0
        for f in latent_samples:
            hidden_samples = self.decoder(f)

            mean_samples_ = self.dec_mean(hidden_samples)
            disp_samples_ = (torch.exp(torch.clamp(self.dec_disp, -15., 15.))).unsqueeze(0)

            mean_samples.append(mean_samples_)
            disp_samples.append(disp_samples_)
            mean_samples_ = mean_samples_.view(y_b,y_num_p,-1).sum(1)


            recon_loss += self.NB_loss(x=raw_y, mean=mean_samples_, disp=disp_samples_, scale_factor=size_factors)
        recon_loss = recon_loss / num_samples

        # ELBO
        elbo = img_recon_loss + recon_loss + self.beta * gp_KL_term + self.beta * gaussian_KL_term

        return elbo, img_recon_loss, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, gp_p_m, gp_p_v, qnet_mu, qnet_var, mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples

    
    def batching_predict_WSI_genes(self, dataset, batch_size=512,n_samples=1):
        self.eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        mean_samples = []

        for batch_idx, (img_batch, loc_batch) in enumerate(dataloader):
            x_batch = loc_batch.to(self.device)
            y_batch = img_batch.to(self.device)
            
            qnet_mu, qnet_var = self.encoder(y_batch)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(x_batch, x_batch, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            mean_samples_ = []
            for f in latent_samples:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            # mean_samples_ = mean_samples_.view(y_b,y_num_p,-1)
            mean_samples.append(mean_samples_.data.cpu().detach())

        mean_samples = torch.cat(mean_samples, dim=0)

        return mean_samples.numpy()

    def batching_predict_train_genes(self, dataset, batch_size=512,n_samples=1):
        self.eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        mean_samples = []

        for batch_idx, (img_batch, loc_batch, cnt_raw_batch, sf_batch) in enumerate(dataloader):
            x_batch = loc_batch.to(self.device)
            y_batch = img_batch.to(self.device)
            y_raw_batch = cnt_raw_batch.to(self.device)
            sf_batch = sf_batch.to(self.device)

            x_b,x_num_p,loc = x_batch.shape
            x_batch = x_batch.reshape(x_b*x_num_p,loc)
            y_b,y_num_p,y_c,h,w = y_batch.shape
            y_batch = y_batch.reshape(y_b*y_num_p,y_c,h,w)
            
            qnet_mu, qnet_var = self.encoder(y_batch)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(x_batch, x_batch, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            mean_samples_ = []
            for f in latent_samples:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples_ = mean_samples_.view(y_b,y_num_p,-1)
            mean_samples.append(mean_samples_.data.cpu().detach())

        mean_samples = torch.cat(mean_samples, dim=0)

        return mean_samples.numpy()
    


    def train_model(self, st_dataset, lr=0.001, weight_decay=0.001, batch_size=256, num_samples=1, 
            train_size=1, maxiter=1000, patience=50, save_model=True, model_weights="model.pt", print_kernel_scale=True):
        """
        Model training.

        """

        self.train()
        log_file = open('Total_log_file.txt',mode='a+')

        if train_size < 1:
            train_dataset, validate_dataset = st_dataset["train"], st_dataset["val"]
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = st_dataset["train"]

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        early_stopping = EarlyStopping(patience=patience, modelfile=model_weights)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        queue = deque()

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            img_recon_loss_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            gaussian_KL_term_val = 0
            num = 0
            for batch_idx, (img_batch, loc_batch, cnt_raw_batch, sf_batch) in enumerate(dataloader):
                x_batch = loc_batch.to(self.device)
                y_batch = img_batch.to(self.device)
                y_raw_batch = cnt_raw_batch.to(self.device)
                sf_batch = sf_batch.to(self.device)

                elbo, img_recon_loss, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples= \
                    self.forward(x=x_batch, y=y_batch, raw_y=y_raw_batch, size_factors=sf_batch, num_samples=num_samples)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                img_recon_loss_val += img_recon_loss.item()
                recon_loss_val += recon_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                gaussian_KL_term_val += gaussian_KL_term.item()

                num += x_batch.shape[0]

                if self.dynamicVAE:
                    KL_val = (gp_KL_term.item() + gaussian_KL_term.item()) / x_batch.shape[0]
                    queue.append(KL_val)
                    avg_KL = np.mean(queue)
                    self.beta, _ = self.PID.pid(self.KL_loss*(self.GP_dim+self.Normal_dim), avg_KL)
                    if len(queue) >= 10:
                        queue.popleft()


            elbo_val = elbo_val/num
            img_recon_loss_val = img_recon_loss_val/num
            recon_loss_val = recon_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            gaussian_KL_term_val = gaussian_KL_term_val/num

            print('Training epoch {}, ELBO:{:.8f}, MSE loss:{:.8f}, NB loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f}'.format(epoch+1, elbo_val, img_recon_loss_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val))
            print('Current beta', self.beta)

            temp = 'Training epoch {}, ELBO:{:.8f}, MSE loss:{:.8f}, NB loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f}, Current beta {}'.format(epoch+1, elbo_val, img_recon_loss_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val, self.beta)
            log_file.write(temp+'\n')
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)
                temp_kernel_scale = 'Current kernel scale {}'.format(torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)
                log_file.write(temp_kernel_scale+'\n')

            if train_size < 1:
                validate_elbo_val = 0
                validate_num = 0
                for _, (validate_img_batch, validate_loc_batch, validate_y_raw_batch, validate_sf_batch) in enumerate(validate_dataloader):
                    validate_x_batch = validate_loc_batch.to(self.device)
                    validate_y_batch = validate_img_batch.to(self.device)
                    validate_y_raw_batch = validate_y_raw_batch.to(self.device)
                    validate_sf_batch = validate_sf_batch.to(self.device)

                    validate_elbo, *_, = \
                        self.forward(x=validate_x_batch, y=validate_y_batch, raw_y=validate_y_raw_batch, size_factors=validate_sf_batch, num_samples=num_samples)

                    validate_elbo_val += validate_elbo.item()
                    validate_num += validate_x_batch.shape[0]

                validate_elbo_val = validate_elbo_val / validate_num

                print("Training epoch {}, validating ELBO:{:.8f}".format(epoch+1, validate_elbo_val))
                early_stopping(validate_elbo_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch+1))
                    break
            
            if train_size == 1:
                early_stopping(elbo_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch+1))
                    break

        if save_model:
            torch.save(self.state_dict(), model_weights)


# if __name__ == '__main__':
#     model = HistoSPAVAE(depth=50, num_classes=5,att_type='dsa')
#     # print(model)
#     input_img = torch.randn(1, 3, 36, 36)
#     input_img = torch.randn(1, 2)
#     out = model(input)
#     # out = model.features(input)
#     print(out.shape)