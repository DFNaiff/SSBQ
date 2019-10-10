# -*- coding: utf-8 -*-
import torch


class Mean(torch.nn.Module):
    """
        Abstract class for mean
    """
    def __init__(self):
        super().__init__()
        
    def mean(self,X):
        raise NotImplementedError

    def integral_mvn(self,mu,Sigma):
        raise NotImplementedError

    def __call__(self,X):
        return self.mean(X)


class FixedConstantMean(Mean):
    """
        Fixed constant mean
    """
    def __init__(self,c=0.0):
        super().__init__()
        self.constant = c
    
    def mean(self,X):
        return self.constant*torch.ones(X.shape[0],1)
    
    def integral_mvn(self,mu,Sigma):
        return torch.tensor([self.constant])


class ZeroMean(FixedConstantMean):
    """
        Zero mean
    """
    def __init__(self):
        super().__init__(0.0)
        

class VariableConstantMean(Mean):
    """
        Fixed constant mean
    """
    def __init__(self,c=0.0):
        super().__init__()
        self.constant = torch.nn.Parameter(torch.tensor(c))
    
    def mean(self,X):
        return self.constant*torch.ones(X.shape[0],1)
    
    def integral_mvn(self,mu,Sigma):
        return torch.tensor([self.constant])
