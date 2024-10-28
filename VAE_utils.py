import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


class DenseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", dropout=0, dtype=torch.float32, norm="batchnorm"):
        super(DenseEncoder, self).__init__()
        self.layers = buildNetwork([input_dim]+hidden_dims, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.enc_mu = nn.Linear(hidden_dims[-1], output_dim)
        self.enc_var = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        h = self.layers(x)
        mu = self.enc_mu(h)
        var = torch.exp(self.enc_var(h).clamp(-15, 15))
        return mu, var

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dims, output_dim, in_channels=32, conv_dims=None, activation="relu", dropout=0, dtype=torch.float32, norm="batchnorm"):
        super(ImageEncoder, self).__init__()
        self.embed_data = nn.Conv2d(3, in_channels, kernel_size=1)
        modules = []
        if conv_dims is None:
            conv_dims = [64, 128, 256]
        for c_dim in conv_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=c_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(c_dim),
                    nn.LeakyReLU())
            )
            in_channels = c_dim

        self.conv_encoder = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layers = buildNetwork([conv_dims[-1]]+hidden_dims, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.enc_mu = nn.Linear(hidden_dims[-1],  output_dim)
        self.enc_var = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.embed_data(x)
        x = self.conv_encoder(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        h = self.layers(x)
        mu = self.enc_mu(h)
        var = torch.exp(self.enc_var(h).clamp(-15, 15))
        return mu, var
    
class ImageDecoder(nn.Module):
    def __init__(self, latent_dims, deconv_dims=None):
        super(ImageDecoder, self).__init__()
        
        modules = []
        if deconv_dims is None:
            deconv_dims = [256, 128, 64]
        self.decoder_input = nn.Linear(latent_dims[-1],  deconv_dims[0] * 16)
        for i in range(len(deconv_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(deconv_dims[i],
                                       deconv_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(deconv_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.conv_decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(deconv_dims[-1],
                                               deconv_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(deconv_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(deconv_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 256,4,4)
        result = self.conv_decoder(result)
        result = self.final_layer(result)
        return result
    


def buildNetwork(layers, network="decoder", activation="relu", dropout=0., dtype=torch.float32, norm="batchnorm"):
    net = []
    if network == "encoder" and dropout > 0:
        net.append(nn.Dropout(p=dropout))
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if norm == "batchnorm":
            net.append(nn.BatchNorm1d(layers[i]))
        elif norm == "layernorm":
            net.append(nn.LayerNorm(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=None):
        eps = 1e-10
        if scale_factor is not None:
            scale_factor = scale_factor[:, None]
            mean = mean * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        log_nb = t1 + t2
#        result = torch.mean(torch.sum(result, dim=1))
        result = torch.sum(log_nb)
        return result

class MixtureNBLoss(nn.Module):
    def __init__(self):
        super(MixtureNBLoss, self).__init__()

    def forward(self, x, mean1, mean2, disp, pi_logits, scale_factor=None):
        eps = 1e-10
        if scale_factor is not None:
            scale_factor = scale_factor[:, None]
            mean1 = mean1 * scale_factor
            mean2 = mean2 * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2_1 = (disp+x) * torch.log(1.0 + (mean1/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean1+eps)))
        log_nb_1 = t1 + t2_1

        t2_2 = (disp+x) * torch.log(1.0 + (mean2/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean2+eps)))
        log_nb_2 = t1 + t2_2

        logsumexp = torch.logsumexp(torch.stack((- log_nb_1, - log_nb_2 - pi_logits)), dim=0)
        softplus_pi = F.softplus(-pi_logits)

        log_mixture_nb = logsumexp - softplus_pi
        result = torch.sum(-log_mixture_nb)
        return result


class PoissonLoss(nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, x, mean, scale_factor=1.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        result = mean - x * torch.log(mean+eps) + torch.lgamma(x+eps)
        result = torch.sum(result)
        return result


def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable
    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = 1.8378770664093453  # log(2*pi)
    term1 = torch.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy


def gmm_fit(data: np.ndarray, mode_coeff=0.6, min_thres=0.3):
    """Returns delta estimate using GMM technique"""
    # Custom definition
    gmm = GaussianMixture(n_components=3)
    gmm.fit(data[:, None])
    vals = np.sort(gmm.means_.squeeze())
    res = mode_coeff * np.abs(vals[[0, -1]]).mean()
    res = np.maximum(min_thres, res)
    return res


# if __name__ == '__main__':
    # input=torch.randn(2,3,32,32)
    # Encoder = ImageEncoder(hidden_dims=[128],output_dim=20)
    # mu,var=Encoder(input)
