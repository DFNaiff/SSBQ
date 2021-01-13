# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../../src')

import math

import torch
import matplotlib.pyplot as plt

from ssgp import SSGP
from base_samplers import RBFSampler,MaternSampler

torch.manual_seed(100)
def sinc_t(x):
    return torch.sin(math.pi*x)/(math.pi*x)
X = torch.rand(20,1)*6-1
Y = sinc_t(X) + 0.05*torch.randn_like(X)

ssgp = SSGP(RBFSampler())
ssgp.set_training_data(X,Y)
ssgp.set_params(1.0*torch.ones(1),
                1.0*torch.ones(1),
                1.0*torch.ones(1),
                100)
optimizer = torch.optim.Adam(ssgp.parameters(),lr=1e-2)

for i in range(500):
    optimizer.zero_grad()
    loss = -ssgp.marginal_log_likelihood()
    loss.backward()
    optimizer.step()
    print(i,loss.item())
    print('--')

X_new = torch.linspace(-1,15,101).reshape(-1,1)
with torch.no_grad():
    Y_new_mean,Y_new_var = ssgp.prediction(X_new)
lower = Y_new_mean - 2*torch.sqrt(Y_new_var)
upper = Y_new_mean + 2*torch.sqrt(Y_new_var)
plt.plot(X.numpy(),Y.numpy(),'ro',alpha=0.5)
plt.plot(X_new.numpy(),Y_new_mean.numpy(),'b-')
plt.fill_between(X_new.flatten().numpy(),
                 lower.flatten().numpy(),
                 upper.flatten().numpy(),alpha=0.8)
plt.plot(X_new.numpy(),sinc_t(X_new).numpy(),color='black',linestyle='--')
#with torch.no_grad():
##    plt.figure()
#    for i in range(100):
#        f = ssgp.sample(1,100)
#        plt.plot(X_new.numpy(),f(X_new).numpy(),'b--',alpha=0.1)