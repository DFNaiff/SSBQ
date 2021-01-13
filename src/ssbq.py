import math

import torch

import utils
import means
import ssgp


class SSBQ(ssgp.SSGP):
    def __init__(self,base_sampler,mean=means.ZeroMean()):
        super().__init__(base_sampler,mean)
        
    def integral_mvn(self,mu,Sigma):
        #mu : (d,1)
        #Sigma : (d,d)
        dim = self.X.shape[1]
        m = self.m
        s0 = self.base_sampler.sample(m,dim) #(m,d)
        s = s0/self.l
        Phi = self._make_phi(s,self.X) #(2m,n)
        L = self._make_gram_cholesky(Phi,m)
        Phi_int = self._make_phi_int(s,mu,Sigma)
        LPhi_int = torch.triangular_solve(Phi_int,L,upper=False)[0]
        PhiY = Phi@self._centered_Y() #(2m,1)
        LPhiY = torch.triangular_solve(PhiY,L,upper=False)[0]
        mean = (LPhi_int*LPhiY).sum() + self.mean.integral_mvn(mu,Sigma)
        var = self.sigma2*(1+(LPhi_int*LPhi_int).sum())
        return mean,var

    def integral_mixmvn(self,mu,Sigma,weights):
        #mu : (t,d,1)
        #Sigma : (t,d,d)
        #weights : (t,1)
        raise NotImplementedError        
        
    def _make_phi_int(self,s,mu,Sigma):
        Phi_int_exp = torch.exp(-0.5*(s*((Sigma@s.t()).t())).sum(dim=-1,keepdim=True)) #(m,1)
        Phi_int_1 = Phi_int_exp*torch.cos((s*mu.t()).sum(dim=1,keepdim=True)) #(m,1)
        Phi_int_2 = Phi_int_exp*torch.sin((s*mu.t()).sum(dim=1,keepdim=True)) #(m,1)
        Phi_int = torch.cat([Phi_int_1,Phi_int_2],dim=1).reshape(-1,Phi_int_1.shape[1]) #(2m,n)
        return Phi_int
    
    