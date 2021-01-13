# -*- coding: utf-8 -*-
import torch

def isnone(x):
    return type(x) == type(None)

def softplus(x):
    return torch.log(1+torch.exp(x))

def inv_softplus(x):
    return torch.log(torch.exp(x)-1)

def jitterize(K,j,proportional=False):
    if proportional:
        if len(K.shape) > 2:
            raise NotImplementedError
        jitter_factor = torch.mean(torch.diag(K)).item()*j
    else:
        jitter_factor = j
    if len(K.shape) > 2:
        if (len(jitter_factor.shape) > 0 and \
            len(jitter_factor.shape) < len(K.shape) - 1):
            jitter_factor = jitter_factor.unsqueeze(-1)
    inds = range(K.shape[-1])
    K[...,inds,inds] += jitter_factor
    return K 

def meshgrid3d(X,Y):
    """
        input:
            X : (m,d) tensor
            Y : (n,d) tensor
        returns:
            Xd : (m x n x d) tensor
            Yd : (m x n x d) tensor
    """
    Xd = X.reshape(X.shape[0],1,X.shape[1])
    Yd = Y.reshape(1,Y.shape[0],Y.shape[1])
    Xd = Xd.repeat([1,Y.shape[0],1])
    Yd = Yd.repeat([X.shape[0],1,1])
    return Xd,Yd
