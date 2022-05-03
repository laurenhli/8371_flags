import numpy as np
from scipy.linalg import norm 
from scipy.stats import unitary_group

class Reg: 
    def __init__(self,n):
        self.n=n
        self.psi=np.zeros((2,)*n) 
        self.psi[(0,)*n]=1
    def rho(self):
        return np.matmul(np.transpose(self.psi),self.psi)

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


def reduced(i, reg):
    """Calculated partial trace to remove flag (qubit i)"""
    np.transpose(reg.psi)
    

def setstate():
    """Calculate initial state"""
    pass

def randstate():
    """Calculate random state"""
