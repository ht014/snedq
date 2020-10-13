from contextlib import contextmanager

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_utils import Lambda, GumbelSoftmax
from .utils import check_numpy


class QuantizationBase(nn.Module):
    def get_codes(self, x):
         
        raise NotImplementedError()

    def get_distances(self, x):
        
        raise NotImplementedError()


class NeuralQuantization(QuantizationBase):
    def __init__(self, vector_dim, num_codebooks=8, codebook_size=256, initial_entropy=1.0, key_dim=None,
                 decouple_temperatures=True, encoder=None, decoder=None, init_codes_with_data=True, **kwargs):
        
        super().__init__()

        key_dim = key_dim or vector_dim

        self.num_codebooks, self.codebook_size = num_codebooks, codebook_size
        self.decouple_temperatures = decouple_temperatures
        self.encoder = encoder or nn.Sequential(
            nn.Linear(vector_dim, num_codebooks * key_dim),
            Lambda(lambda x: x.view(*x.shape[:-1], num_codebooks, key_dim)),
        )
        self.codebook = nn.Parameter(torch.randn(num_codebooks, codebook_size, key_dim))
        self.decoder = decoder or nn.Sequential(
            Lambda(lambda x: x.view(*x.shape[:-2], -1)),
            nn.Linear(num_codebooks * codebook_size, vector_dim),
        )
        self.log_temperatures = nn.Parameter(data=torch.zeros(num_codebooks) * float('nan'), requires_grad=True)
        self.initial_entropy, self.init_codes_with_data = initial_entropy, init_codes_with_data
        self.gumbel_softmax = GumbelSoftmax(**kwargs)

    def compute_logits(self, x, add_temperatures=True):
        
        assert len(x.shape) >= 2, "x should be of shape [..., vector_dim]"
        if len(x.shape) > 2:
            flat_logits = self.compute_logits(x.view(-1, x.shape[-1]), add_temperatures=add_temperatures)
            return flat_logits.view(*x.shape[:-1], self.num_codebooks, self.codebook_size)

        # einsum: [b]atch_size, [n]um_codebooks, [c]odebook_size, [v]ector_dim
        logits = torch.einsum('bnd,ncd->bnc', self.encoder(x), self.codebook)

        if add_temperatures:
            if not self.is_initialized(): self.initialize(x)
            logits *= torch.exp(-self.log_temperatures[:, None])
        return logits

    def forward(self, x, return_intermediate_values=False):
        
        if not self.is_initialized(): self.initialize(x)
        logits_raw = self.compute_logits(x, add_temperatures=False)
        logits = logits_raw * torch.exp(-self.log_temperatures[:, None])
        codes = self.gumbel_softmax(logits, dim=-1)  # [..., num_codebooks, codebook_size]
        x_reco = self.decoder(codes)

        if return_intermediate_values:
            distances_to_codes = - (logits_raw if self.decouple_temperatures else logits)
            return x_reco, dict(x=x, logits=logits, codes=codes, x_reco=x_reco,
                                distances_to_codes=distances_to_codes)
        else:
            return x_reco

    def get_codes(self, x):
        
        return self.compute_logits(x, add_temperatures=False).argmax(dim=-1)

    def get_distances(self, x):
        # Note: this quantizer uses the fact that logits = - distances
        return - self.compute_logits(x, add_temperatures=not self.decouple_temperatures)

    def is_initialized(self):
     
        return check_numpy(torch.isfinite(self.log_temperatures.data)).all()

    def initialize(self, x):
       
        with torch.no_grad():
            if self.init_codes_with_data:
                chosen_ix = torch.randint(0, x.shape[0], size=[self.codebook_size * self.num_codebooks], device=x.device)
                chunk_ix = torch.arange(self.codebook_size * self.num_codebooks, device=x.device) // self.codebook_size
                
                initial_keys = self.encoder(x)[chosen_ix, chunk_ix].view(*self.codebook.shape).contiguous()
                self.codebook.data[:] = initial_keys

            base_logits = self.compute_logits(
                x, add_temperatures=False).view(-1, self.num_codebooks, self.codebook_size)
            

            log_temperatures = torch.tensor([
                fit_log_temperature(codebook_logits, target_entropy=self.initial_entropy, tolerance=1e-2)
                for codebook_logits in check_numpy(base_logits).transpose(1, 0, 2)
            ], device=x.device, dtype=x.dtype)
            self.log_temperatures.data[:] = log_temperatures


def fit_log_temperature(logits, target_entropy=1.0, tolerance=1e-6, max_steps=100,
                        lower_bound=math.log(1e-9), upper_bound=math.log(1e9)):
    
    log_tau = (lower_bound + upper_bound) / 2.0

    for i in range(max_steps):
        # check temperature at the geometric mean between min and max values
        log_tau = (lower_bound + upper_bound) / 2.0
        tau_entropy = _entropy_with_logits(logits, log_tau)

        if abs(tau_entropy - target_entropy) < tolerance:
            break
        elif tau_entropy > target_entropy:
            upper_bound = log_tau
        else:
            lower_bound = log_tau
    return log_tau


def _entropy_with_logits(logits, log_tau=0.0, axis=-1):
    logits = np.copy(logits)
    logits -= np.max(logits, axis, keepdims=True)
    logits *= np.exp(-log_tau)
    exps = np.exp(logits)
    sum_exp = exps.sum(axis)
    entropy_values = np.log(sum_exp) - (logits * exps).sum(axis) / sum_exp
    return np.mean(entropy_values)


def compute_penalties(logits, individual_entropy_coeff=0.0, allowed_entropy=0.0, global_entropy_coeff=0.0,
                      cv_coeff=0.1, square_cv=True, eps=1e-9):
    
    counters = dict(reg=torch.tensor(0.0, dtype=torch.float32, device=logits.device))
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    
    if individual_entropy_coeff != 0:
        individual_entropy_values = - torch.sum(p * logp, dim=-1)
        clipped_entropy = F.relu(allowed_entropy - individual_entropy_values + eps).mean()
        individual_entropy = (individual_entropy_values.mean() - clipped_entropy).detach() + clipped_entropy

        counters['reg'] += individual_entropy_coeff * individual_entropy
      

    if global_entropy_coeff != 0:
        global_p = torch.mean(p, dim=0)  # [..., codebook_size]
        global_logp = torch.logsumexp(logp, dim=0) - np.log(float(logp.shape[0]))  # [..., codebook_size]
        global_entropy = - torch.sum(global_p * global_logp, dim=-1).mean()
        counters['reg'] += global_entropy_coeff * global_entropy
        

    

    return counters
