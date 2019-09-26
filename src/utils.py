# -*- coding: utf-8 -*-
import torch


def softplus(x):
    return torch.log(1+torch.exp(x))

def inv_softplus(x):
    return torch.log(torch.exp(x)-1)

def jitterize(K,j,proportional=False):
    if proportional:
        jitter_factor = torch.mean(torch.diag(K)).item()*j
    else:
        jitter_factor = j
    K[range(len(K)),range(len(K))] += jitter_factor
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
