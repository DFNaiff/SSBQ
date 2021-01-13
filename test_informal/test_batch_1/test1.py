# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../../src')

import math

import torch
import matplotlib.pyplot as plt

from ssgp import SSGP
from ssbq import SSBQ
from base_samplers import RBFSampler,MaternSampler

X = torch.randn(10000).reshape(-1,1)
Y = torch.sin(0.5*math.pi*X) + 0.001*torch.randn_like(X)

ssgp = SSBQ(MaternSampler(1.5))
ssgp.set_training_data(X,Y)
ssgp.set_params(1.0*torch.ones(1),
                1.0*torch.ones(1),
                1.0*torch.ones(1),
                50)
optimizer = torch.optim.Adam(ssgp.parameters(),lr=1e-2)

for i in range(500):
    optimizer.zero_grad()
    loss = -ssgp.marginal_log_likelihood()
    loss.backward()
    optimizer.step()
    print(i,loss.item())
    print('--')

X_new = torch.linspace(-5,5,51).reshape(-1,1)
with torch.no_grad():
    Y_new_mean,Y_new_var = ssgp.prediction(X_new)
lower = Y_new_mean - 2*torch.sqrt(Y_new_var)
upper = Y_new_mean + 2*torch.sqrt(Y_new_var)
plt.plot(X.numpy(),Y.numpy(),'ro',alpha=0.01)
plt.plot(X_new.numpy(),Y_new_mean.numpy(),'b-')
plt.fill_between(X_new.flatten().numpy(),
                 lower.flatten().numpy(),
                 upper.flatten().numpy(),alpha=0.8)
ssgp.integral_mvn(torch.zeros(1,1),torch.ones(1,1))
