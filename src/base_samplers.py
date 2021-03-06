# -*- coding: utf-8 -*-
import math

import torch

class BaseSampler(object):
    """
        Abstract class describing all samplers
    """
    def sample(self,*shape):
        raise NotImplementedError
        

class RBFSampler(BaseSampler):
    """
        Spectral density distribution of base RBF kernel
        Is a 1/math.pi scaled normal distribution
    """
    def __init__(self):
        pass
    
    def sample(self,*shape):
        return torch.randn(*shape)/math.pi


class MaternSampler(BaseSampler):
    """
        Spectral density distribution of base Matern kernel
        Is a 1/(2*math.pi) scaled 2*nu-dof multivariate t-distribution
    """
    def __init__(self,nu):
        self.nu = nu
        
    def sample(self,*shape):
        n = shape[-2]
        y = torch.randn(*shape)
        u = torch.randn(n,int(2*self.nu)).pow(2).sum(dim=1,keepdim=True)
        x = y/torch.sqrt(u/(2*self.nu))
        return x/(2*math.pi)
        
        
        