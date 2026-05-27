from __future__ import annotations
from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast
from einops import rearrange, pack, unpack

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))



class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, online_clustered: bool=True):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

        # Used to record the utilization of code vectors in the codebook during inference
        self.used_count = torch.zeros(self.codebook_size)
        self.reset_used_count = False

        self.batch_count = 1
        self.reset_batch_count = False
        self.avg_probs_all = 0.0
        
        # Used to store the usage probability of each vector in the codebook
        self.register_buffer("embed_prob", torch.zeros(self.codebook_size))
        self.decay = 0.99
        self.anchor = 'probrandom' # or 'probrandom', 'random', 'closest'
        self.pool = FeaturePool(self.codebook_size, self.codebook_dim)

        self.contras_loss = False
        self.balancing_loss = False
        self.online_clustered = online_clustered
        

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        
        # Encoding and Quantise
        z_q, indices, perplexity, utilization_rates, contra_loss, balancing_loss = self.decode_latents(z_e)

        # Compute loss for embedding
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        # Preserve gradients
        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, rearrange(z_e, "b d t -> (b t) d"), perplexity, utilization_rates, contra_loss, balancing_loss

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )

        # Encoding
        codewords = (-dist).max(1)[1]

        # Analyze codebook
        # Calculate codebook utilization rate
        encodings_indices = torch.zeros(encodings.shape[0], self.codebook_size, device=latents.device)
        encodings_indices.scatter_(1, codewords.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings_indices, dim=0)
        # For each batch
        batch_used_count = (avg_probs > 0).sum().item()
        batch_utilization_rate = batch_used_count / self.codebook_size
        # For all the batches
        if self.training:
            self.reset_used_count = True
        if (not self.training) and self.reset_used_count:
            self.used_count = torch.zeros(self.codebook_size)
            self.reset_used_count = False
        batch_used_indices = torch.nonzero(avg_probs > 0).squeeze()
        self.used_count = self.used_count.to(latents.device)
        self.used_count[batch_used_indices] += 1
        utilization_rate = (self.used_count > 0).sum().item() / self.codebook_size

        # Calculate codebook perplexity
        # For each batch
        batch_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # For all the batches
        if self.training:
            self.reset_batch_count = True
        if (not self.training) and self.reset_batch_count:
            self.batch_count = 1
            self.reset_batch_count = False
        self.avg_probs_all = (self.batch_count-1)/self.batch_count*self.avg_probs_all + 1/self.batch_count*avg_probs
        perplexity = torch.exp(-torch.sum(self.avg_probs_all * torch.log(self.avg_probs_all + 1e-10)))
        self.batch_count += 1

        # Quantise
        codewords = rearrange(codewords, "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(codewords)

        # online clustered reinitialisation for unoptimized points
        if self.online_clustered and self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # closest sampling
            if self.anchor == 'closest':
                sort_distance, indices = dist.sort(dim=0)
                random_feat = encodings.detach()[indices[-1,:]]
            # probabilitical based random sampling
            elif self.anchor == 'probrandom':
                norm_distance = F.softmax(dist.t(), dim=1)
                prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                random_feat = encodings.detach()[prob]
            # feature pool based random sampling
            elif self.anchor == 'random':
                random_feat = self.pool.query(encodings.detach())
            # decay parameter based on the average usage
            decay = torch.exp(-(self.embed_prob*self.codebook_size*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.codebook_dim)
            self.codebook.weight.data = self.codebook.weight.data * (1 - decay) + random_feat * decay
        
        # contrastive loss
        if self.contras_loss:
            sort_distance, indices = dist.sort(dim=0)
            dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.codebook_size)):,:].mean(dim=0, keepdim=True)
            dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
            dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
            contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
        else:
            contra_loss = torch.zeros(1).to(z_q.device)

        # codevector balancing loss
        if self.balancing_loss:
            balancing_loss = F.binary_cross_entropy(avg_probs, torch.full_like(avg_probs, 1.0 / self.codebook_size))
            # balancing_loss = F.kl_div(torch.log(avg_probs), torch.full_like(avg_probs, 1.0 / self.codebook_size), reduction='batchmean')
        else:
            balancing_loss = torch.zeros(1).to(z_q.device)


        return z_q, codewords, (batch_perplexity, perplexity), (batch_utilization_rate, utilization_rate), contra_loss, balancing_loss


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        online_clustered = [True]

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i], online_clustered=online_clustered[i])
                for i in range(0, n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

        self.ssim_loss = False

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        contra_loss = 0
        balancing_loss = 0

        codebook_indices = []
        latents = []
        z_q_is= []
        
        # store codebook utilization rates of each quantizer
        batch_utilzation_rates = []
        utilzation_rates = []
        batch_perplexities = []
        perplexities = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i, perplexities_i, utilization_rates_i, contra_loss_i, balancing_loss_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            z_q_is.append(z_q_i * mask[:, None, None])
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()
            contra_loss += (contra_loss_i * mask).mean()
            balancing_loss += (balancing_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)
            batch_utilzation_rates.append(utilization_rates_i[0])
            utilzation_rates.append(utilization_rates_i[1])
            batch_perplexities.append(perplexities_i[0])
            perplexities.append(perplexities_i[1])

        codes = torch.stack(codebook_indices, dim=1)
        # latents = torch.cat(latents, dim=1)
        latents = torch.stack(latents)

        if self.ssim_loss:
            ssim_loss_vq12 = ssim(z_q_is[0], z_q_is[1])
            ssim_loss_vq23 = ssim(z_q_is[1], z_q_is[2])
            ssim_loss_vq34 = ssim(z_q_is[2], z_q_is[3])
            ssim_loss = ssim_loss_vq12 + ssim_loss_vq23 + ssim_loss_vq34
        else:
            ssim_loss = (torch.zeros(1).to(latents.device) * mask).mean()

        return z_q, codes, latents, commitment_loss, codebook_loss, (batch_perplexities, perplexities), (batch_utilzation_rates, utilzation_rates), contra_loss, balancing_loss, ssim_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i, _, _, _ = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = min(self.nums_features + features.size(0), self.pool_size)
                # print(self.nums_features, num, features.shape)
                self.features[self.nums_features:num] = features[:(num-self.nums_features)]
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)
    (_, channel, _, _) = img1.size()
    global window
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
    

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices = True,
        force_quantization_f32 = True
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)
        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        # print(has_projections)
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.type(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices
