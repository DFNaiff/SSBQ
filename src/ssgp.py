import math
import functools

import torch

import utils
import means


class SSGP(torch.nn.Module):
    def __init__(self,base_sampler,mean=means.ZeroMean()):
        super().__init__()
        self.base_sampler = base_sampler
        self.mean = mean
        
    def set_params(self,theta0,l,sigma2,m=50):
        self._raw_theta0 = torch.nn.Parameter(utils.inv_softplus(theta0))
        self._raw_l = torch.nn.Parameter(utils.inv_softplus(l))
        self._raw_sigma2 = torch.nn.Parameter(utils.inv_softplus(sigma2))
        self.m = m
        
    def set_training_data(self,X,Y):
        self.X = X
        self.Y = Y
        self.mean_X = self.mean(X)
        
    def marginal_log_likelihood(self,m=None):
        dim = self.X.shape[1]
        n = self.X.shape[0]
        m = m if not utils.isnone(m) else self.m
        s0 = self.base_sampler.sample(m,dim) #(m,d)
        s = s0/self.l
        Phi = self._make_phi(s,self.X) #(2m,n)
        L = self._make_gram_cholesky(Phi)
        PhiY = Phi@self._centered_Y() #(2m,1)
        LPhiY = torch.triangular_solve(PhiY,L,upper=False)[0]
        term1 = -0.5*torch.sum(self._centered_Y()**2)/self.sigma2
        term2 = 0.5*torch.sum(LPhiY**2)/self.sigma2
        term3 = -torch.sum(torch.log(torch.diag(L)))
        term4 = m*torch.log(m*self.sigma2/self.theta0)
        term5 = -0.5*n*torch.log(2*math.pi*self.sigma2)
        return term1 + term2 + term3 + term4 + term5
    
    def prediction(self,X_new,m=None):
        dim = self.X.shape[1]
        m = m if not utils.isnone(m) else self.m
        s0 = self.base_sampler.sample(m,dim) #(m,d)
        s = s0/self.l
        Phi = self._make_phi(s,self.X) #(2m,n)
        L = self._make_gram_cholesky(Phi)
        Phi_new = self._make_phi(s,X_new) #(2m,n)
        PhiY = Phi@self._centered_Y() #(2m,1)
        LPhiY = torch.triangular_solve(PhiY,L,upper=False)[0]
        LPhi_new = torch.triangular_solve(Phi_new,L,upper=False)[0]
        mean = LPhi_new.t()@LPhiY
        var = self.sigma2*(1+LPhi_new.t().pow(2).sum(dim=-1,keepdim=True))
        return mean,var
        
    def sample(self,dim,m=None):
        """
            Sample function (from basis)
        """
        m = m if not utils.isnone(m) else self.m
        s0 = self.base_sampler.sample(m,dim) #(m,d)
        s = s0/self.l #(m,d)
        a = torch.randn(m)*torch.sqrt(self.theta0/self.m) #(m,)
        b = torch.randn(m)*torch.sqrt(self.theta0/self.m) #(m,)
        return functools.partial(sample_fn,a=a,b=b,s=s)
    
    @property
    def l(self):
        return utils.softplus(self._raw_l)
    
    @property
    def theta0(self):
        return utils.softplus(self._raw_theta0)
    
    @property
    def sigma2(self):
        return utils.softplus(self._raw_sigma2)

    def _make_gram_cholesky(self,Phi):
        A_ = Phi@Phi.t() #(2m,2m)
        A = utils.jitterize(A_,self.m*self.sigma2/self.theta0)
        L = torch.cholesky(A,upper=False)
        return L
    
    def _make_phi(self,s,X):
        Phi_1 = torch.cos(2*math.pi*(s@X.t())) #(m,n)
        Phi_2 = torch.sin(2*math.pi*(s@X.t())) #(m,n)
        Phi = torch.cat([Phi_1,Phi_2],dim=1).reshape(-1,Phi_1.shape[1]) #(2m,n)
        return Phi

    def _centered_Y(self):
        return self.Y - self.mean_X


def sample_fn(x,a,b,s):
    #x : (*,d)
    #returns : (*,1)
    cos_vec = torch.cos(2*math.pi*x@s.t()) #(*,m)
    sin_vec = torch.sin(2*math.pi*x@s.t()) #(*,m)
    res_ = cos_vec*a + sin_vec*b
    res = res_.sum(dim=-1,keepdim=True)
    return res