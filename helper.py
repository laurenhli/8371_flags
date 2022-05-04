import numpy as np
from scipy.linalg import norm, sqrtm
from scipy.stats import unitary_group

class Reg: 
    def __init__(self,n):
        self.n=n
        self.psi=np.zeros((2,)*n) 
        self.psi[(0,)*n]=1
    def rho(self):
        return np.outer(self.psi,self.psi)

X_matrix=np.array([[0, 1],
                    [1,0]])

Y_matrix=np.array([[0, -1j],
                    [1j,0]])

Z_matrix=np.array([[1, 0],
                    [0,-1]])

H_matrix=np.array([[1, 1],
                    [1,-1]])/np.sqrt(2)

def Rn_matrix(n):
    return np.array([[1, 0],
                    [0,np.exp(1j*np.pi/(2**(n-1)))]])

CNOT_matrix=np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,0,1],
                      [0,0,1,0]])

CNOT_tensor=np.reshape(CNOT_matrix, (2,2,2,2))

CZ_matrix=np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,-1]])

CZ_tensor=np.reshape(CZ_matrix, (2,2,2,2))

def CRn_tensor(n):
    A = np.eye(2)
    B = np.zeros((2,2))
    CRn_matrix = np.block([[A,B],
                          [B,Rn_matrix(n)]])
    return np.reshape(CRn_matrix, (2,2,2,2))

def CUrand_tensor(dim):
    total_dim = 2**dim
    block_dim = total_dim/2
    A = np.eye(block_dim)
    B = np.zeros((block_dim,block_dim))
    U = unitary_group.rvs(block_dim)
    CUrand_matrix = np.block([[A,B],
                          [B,U]])
    return np.reshape(CUrand_matrix, (2,)*total_dim)

def X(i,reg): 
    reg.psi=np.tensordot(X_matrix,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def Y(i,reg): 
    reg.psi=np.tensordot(Y_matrix,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def Z(i,reg): 
    reg.psi=np.tensordot(Z_matrix,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)
        
def H(i,reg): 
    reg.psi=np.tensordot(H_matrix,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def Rn(i,n,reg): 
    reg.psi=np.tensordot(Rn_matrix(n),reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def CNOT(control, target, reg):
    reg.psi=np.tensordot(CNOT_tensor, reg.psi, ((2,3),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target))   

def CZ(control, target, reg):
    reg.psi=np.tensordot(CZ_tensor, reg.psi, ((2,3),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target))  

def CRn(control, target, n, reg):
    reg.psi=np.tensordot(CRn_tensor(n), reg.psi, ((2,3),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target))   

def CUrand(control, target, dim, reg):
    total_dim = 2**dim
    reg.psi=np.tensordot(CUrand_tensor(dim), reg.psi, ((total_dim-2,total_dim-1),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target)) 

def measure(i, reg): 
    z_proj=[np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]])] 
    x_proj=[np.array([[1,1],[1,1]])/2, np.array([[1,-1],[-1,1]])/2]
    
    def project(i,j,reg): 
        projected=np.tensordot(z_proj[j],reg.psi,(1,i))
        return np.moveaxis(projected,0,i)
        #if basis=='z':
        #    projected=np.tensordot(z_proj[j],reg.psi,(1,i))
        #elif basis=='x':
        #    projected=np.tensordot(x_proj[j],reg.psi,(1,i))
        #return np.moveaxis(projected,0,i)
    
    projected=project(i,0,reg) 
    norm_projected=norm(projected.flatten()) 
    if np.random.random()<norm_projected**2: 
        reg.psi=projected/norm_projected
        return 0
    else:
        projected=project(i,1,reg)
        reg.psi=projected/norm(projected)
        return 1


def reducedrho(qb, reg):
    """Calculated partial trace to remove flag (qubit i)"""
    moved_psi = reg.psi
    for i in sorted(qb, reverse=True):
        moved_psi = np.moveaxis(moved_psi,i,0)
    dim = np.size(reg.psi)
    reddim = 2**len(qb)
    rho = np.outer(moved_psi, moved_psi)
    reshaped_rho = np.reshape(rho, [reddim, int(dim/reddim), reddim, int(dim/reddim)])
    return np.einsum('ijik->jk', reshaped_rho)
    #moved_psi = np.moveaxis(reg.psi,i,0)
    #dim = np.size(reg.psi)
    #rho = np.outer(moved_psi, moved_psi)
    #reshaped_rho = np.reshape(rho, [2, int(dim/2), 2, int(dim/2)])
    #return np.einsum('ijik->jk', reshaped_rho)
    
def setstate(D, reg):
    """Calculate initial state
    D = dictionary {(position): val}
    """
    for pos, val in D.items():
        reg.psi[pos] = val

def randstate(nflags, reg):
    """Calculate random state"""
    dim = np.size(reg.psi) / (2**nflags)
    raw = np.random.uniform(0,1,dim)
    denom = np.sqrt(np.sum(np.square(raw)))
    D = {}
    for i in dim:
        bin =  list(np.binary_repr(i))
        binint = tuple([int(bit) for bit in bin])
        D[(0,)*nflags+binint] = raw[i]/denom
    setstate(D, reg)


def fidelity(rho1, rho2):
    rho1_sqrt = sqrtm(rho1)
    prod = np.matmul(rho1_sqrt,(np.matmul(rho2, rho1_sqrt)))
    return np.real(np.trace(sqrtm(prod)))
