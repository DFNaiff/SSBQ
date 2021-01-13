# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../../src')

import math

import torch
import matplotlib.pyplot as plt

from ssgp import SSGP
import utils
from base_samplers import RBFSampler,MaternSampler

torch.manual_seed(100)
def sinc_t(x):
    return torch.sin(math.pi*x)/(math.pi*x)
X = torch.rand(100,1)*6-1
Y = sinc_t(X.sum(dim=1,keepdim=True))
Y += 0.05*torch.randn_like(Y)

ssgp = SSGP(RBFSampler())
ssgp.set_training_data(X,Y)
ssgp.set_params(1.0*torch.ones(1),
                1.0*torch.ones(1),
                1.0*torch.ones(1),
                100)

def elbo(mu,sigma):
    d = mu.shape[0]
    Z = torch.randn(1000,d)
    X = sigma*Z + mu
    X_= utils.softplus(X)
    ll = ssgp._raw_marginal_log_likelihood(X_[:,0],X_[:,1:-1],X_[:,-1],
                                            25) #(k,)
    entropy = d/2*math.log(2*math.pi*math.e) + torch.sum(torch.log(sigma))
    logprior = -0.5*(((X/10.0).pow(2).sum(dim=-1))) #(k,)
    return torch.mean(ll+logprior+entropy)
    
mu,sigma = torch.zeros(3,requires_grad=True),torch.ones(3,requires_grad=True)
optimizer = torch.optim.Adam((mu,sigma),lr=1e-3)
for i in range(1000):
    loss = -elbo(mu,sigma)
    loss.backward()
    optimizer.step()
    print(i,loss)

#with torch.no_grad():
##    plt.figure()
#    for i in range(100):
#        f = ssgp.sample(1,100)
#        plt.plot(X_new.numpy(),f(X_new).numpy(),'b--',alpha=0.1)