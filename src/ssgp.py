import math

import torch
import matplotlib.pyplot as plt

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


class SSGPBQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def set_params(self,theta0,l,sigma2,m=50):
        self._raw_theta0 = torch.nn.Parameter(inv_softplus(theta0))
        self._raw_l = torch.nn.Parameter(inv_softplus(l))
        self._raw_sigma2 = torch.nn.Parameter(inv_softplus(sigma2))
        self.m = m
        
    def set_training_data(self,X,Y):
        self.X = X
        self.Y = Y
        
    def marginal_log_likelihood(self):
        dim = self.X.shape[1]
        n = self.X.shape[0]
        m = self.m
        s0 = torch.randn(m,dim) #(m,d)
        s = s0/(math.pi*self.l)
        Phi_1 = torch.cos(2*math.pi*(s@self.X.t())) #(m,n)
        Phi_2 = torch.sin(2*math.pi*(s@self.X.t())) #(m,n)
        Phi = torch.cat([Phi_1,Phi_2],dim=1).reshape(-1,Phi_1.shape[1]) #(2m,n)
        A_ = Phi@Phi.t() #(2m,2m)
        A = jitterize(A_,m*self.sigma2/self.theta0)
        L = torch.cholesky(A,upper=False)
    #    print(Phi.shape)
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
        n = self.X.shape[0]
        m = self.m
        s0 = torch.randn(m,dim) #(m,d)
        s = s0/(math.pi*self.l)
        Phi_1 = torch.cos(2*math.pi*(s@self.X.t())) #(m,n)
        Phi_2 = torch.sin(2*math.pi*(s@self.X.t())) #(m,n)
        Phi = torch.cat([Phi_1,Phi_2],dim=1).reshape(-1,Phi_1.shape[1]) #(2m,n)
        A_ = Phi@Phi.t() #(2m,2m)
        A = jitterize(A_,m*self.sigma2/self.theta0)
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

    def integral_mvn(self,mu,Sigma):
        #mu : (d,1)
        #Sigma : (d,d)
        dim = self.X.shape[1]
        n = self.X.shape[0]
        m = self.m
        s0 = torch.randn(m,dim) #(m,d)
        s = s0/(math.pi*self.l)
        Phi_1 = torch.cos(2*math.pi*(s@self.X.t())) #(m,n)
        Phi_2 = torch.sin(2*math.pi*(s@self.X.t())) #(m,n)
        Phi = torch.cat([Phi_1,Phi_2],dim=1).reshape(-1,Phi_1.shape[1]) #(2m,n)
        A_ = Phi@Phi.t() #(2m,2m)
        A = jitterize(A_,m*self.sigma2/self.theta0)
        L = torch.cholesky(A,upper=False)
        Phi_int_exp = torch.exp(-0.5*(s*((Sigma@s.t()).t())).sum(dim=-1,keepdim=True)) #(m,1)
        Phi_int_1 = Phi_int_exp*torch.cos((s*mu.t()).sum(dim=1,keepdim=True)) #(m,1)
        Phi_int_2 = Phi_int_exp*torch.cos((s*mu.t()).sum(dim=1,keepdim=True)) #(m,1)
        Phi_int = torch.cat([Phi_int_1,Phi_int_2],dim=1).reshape(-1,Phi_int_1.shape[1]) #(2m,n)
        LPhi_int = torch.triangular_solve(Phi_int,L,upper=False)[0]
        PhiY = Phi@self.Y #(2m,1)
        LPhiY = torch.triangular_solve(PhiY,L,upper=False)[0]
        mean = (LPhi_int*LPhiY).sum()
        var = self.sigma2*(1+(LPhi_int*LPhi_int).sum())
        return mean,var
        
        
    @property
    def l(self):
        return softplus(self._raw_l)
    
    @property
    def theta0(self):
        return softplus(self._raw_theta0)
    
    @property
    def sigma2(self):
        return softplus(self._raw_sigma2)


if __name__ == "__main__":
    X = torch.randn(1000).reshape(-1,1)
    Y = torch.sin(0.5*math.pi*X) + 0.001*torch.randn_like(X)
    
    ssgp = SSGPBQ()
    ssgp.set_training_data(X,Y)
    ssgp.set_params(1.0*torch.ones(1),
                    1.0*torch.ones(1),
                    1.0*torch.ones(1),
                    50)
    optimizer = torch.optim.Adam(ssgp.parameters(),lr=1e-2)
    
    for i in range(1000):
        optimizer.zero_grad()
        loss = -ssgp.marginal_log_likelihood()
        loss.backward()
        optimizer.step()
        print(i,loss.item())
        print('--')
    
    X_new = torch.linspace(-2,2,51).reshape(-1,1)
    with torch.no_grad():
        Y_new_mean,Y_new_var = ssgp.prediction(X_new)
    lower = Y_new_mean - 2*torch.sqrt(Y_new_var)
    upper = Y_new_mean + 2*torch.sqrt(Y_new_var)
    plt.plot(X.numpy(),Y.numpy(),'ro',alpha=0.01)
    plt.plot(X_new.numpy(),Y_new_mean.numpy(),'b-')
    plt.fill_between(X_new.flatten().numpy(),
                     lower.flatten().numpy(),
                     upper.flatten().numpy(),alpha=0.8)
    
    
    
    
