import math

import torch
import matplotlib.pyplot as plt

import utils


class SSGP(torch.nn.Module):
    def __init__(self,base_sampler):
        super().__init__()
        self.base_sampler = base_sampler
        
    def set_params(self,theta0,l,sigma2,m=50):
        self._raw_theta0 = torch.nn.Parameter(utils.inv_softplus(theta0))
        self._raw_l = torch.nn.Parameter(utils.inv_softplus(l))
        self._raw_sigma2 = torch.nn.Parameter(utils.inv_softplus(sigma2))
        self.m = m
        
    def set_training_data(self,X,Y):
        self.X = X
        self.Y = Y
        
    def marginal_log_likelihood(self):
        dim = self.X.shape[1]
        n = self.X.shape[0]
        m = self.m
        s0 = self.base_sampler.sample(m,dim) #(m,d)
        s = s0/self.l
        Phi_1 = torch.cos(2*math.pi*(s@self.X.t())) #(m,n)
        Phi_2 = torch.sin(2*math.pi*(s@self.X.t())) #(m,n)
        Phi = torch.cat([Phi_1,Phi_2],dim=1).reshape(-1,Phi_1.shape[1]) #(2m,n)
        A_ = Phi@Phi.t() #(2m,2m)
        A = utils.jitterize(A_,m*self.sigma2/self.theta0)
        L = torch.cholesky(A,upper=False)
        PhiY = Phi@self.Y #(2m,1)
        LPhiY = torch.triangular_solve(PhiY,L,upper=False)[0]
        term1 = -0.5*torch.sum(self.Y**2)/self.sigma2
        term2 = 0.5*torch.sum(LPhiY**2)/self.sigma2
        term3 = -torch.sum(torch.log(torch.diag(L)))
        term4 = m*torch.log(m*self.sigma2/self.theta0)
        term5 = -0.5*n*torch.log(2*math.pi*self.sigma2)
        return term1 + term2 + term3 + term4 + term5
    
    def prediction(self,X_new):
        dim = self.X.shape[1]
        m = self.m
        s0 = self.base_sampler.sample(m,dim) #(m,d)
        s = s0/self.l
        Phi_1 = torch.cos(2*math.pi*(s@self.X.t())) #(m,n)
        Phi_2 = torch.sin(2*math.pi*(s@self.X.t())) #(m,n)
        Phi = torch.cat([Phi_1,Phi_2],dim=1).reshape(-1,Phi_1.shape[1]) #(2m,n)
        A_ = Phi@Phi.t() #(2m,2m)
        A = utils.jitterize(A_,m*self.sigma2/self.theta0)
        L = torch.cholesky(A,upper=False)
        Phi_new_1 = torch.cos(2*math.pi*s@X_new.t())
        Phi_new_2 = torch.sin(2*math.pi*s@X_new.t())
        Phi_new = torch.cat([Phi_new_1,Phi_new_2],dim=1).\
                    reshape(-1,Phi_new_1.shape[1])
        PhiY = Phi@self.Y #(2m,1)
        LPhiY = torch.triangular_solve(PhiY,L,upper=False)[0]
        LPhi_new = torch.triangular_solve(Phi_new,L,upper=False)[0]
        mean = LPhi_new.t()@LPhiY
        var = self.sigma2*(1+LPhi_new.t().pow(2).sum(dim=-1,keepdim=True))
        return mean,var
        
    @property
    def l(self):
        return utils.softplus(self._raw_l)
    
    @property
    def theta0(self):
        return utils.softplus(self._raw_theta0)
    
    @property
    def sigma2(self):
        return utils.softplus(self._raw_sigma2)

    def _make_gram_cholesky(self,s):
        Phi_1 = torch.cos(2*math.pi*(s@self.X.t())) #(m,n)
        Phi_2 = torch.sin(2*math.pi*(s@self.X.t())) #(m,n)
        Phi = torch.cat([Phi_1,Phi_2],dim=1).reshape(-1,Phi_1.shape[1]) #(2m,n)
        A_ = Phi@Phi.t() #(2m,2m)
        A = utils.jitterize(A_,self.m*self.sigma2/self.theta0)
        L = torch.cholesky(A,upper=False)
        return L
    
