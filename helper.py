import numpy as np
from scipy.linalg import norm, sqrtm
from scipy.stats import unitary_group

class Reg: 
    def __init__(self,n):
        self.n=n
        self.psi=np.zeros((2,)*n) 
        self.psi[(0,)*n]=1
    def rho(self):
        return np.outer(np.conj(self.psi),self.psi)

I_matrix=np.eye(2)

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

def Rndag_matrix(n):
    return np.array([[1, 0],
                    [0,np.exp(-1j*np.pi/(2**(n-1)))]])
def CRn_matrix(n):
    A = np.eye(2)
    B = np.zeros((2,2))
    return np.block([[A,B], [B,Rn_matrix(n)]])

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

def CRndag_tensor(n):
    A = np.eye(2)
    B = np.zeros((2,2))
    CRn_matrix = np.block([[A,B],
                          [B,Rndag_matrix(n)]])
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

def Ccustom2_tensor(custom):
    """custom is 2x2 matrix"""
    dim = 2**2
    block_dim = int(dim/2)
    A = np.eye(block_dim)
    B = np.zeros((block_dim,block_dim))
    Ccustom2_matrix = np.block([[A,B],[B, custom]])
    return np.reshape(Ccustom2_matrix, (2,)*dim)

def Ccustom3_tensor(custom):
    """custom is 4x4 matrix"""
    dim = 2**3
    block_dim = int(dim/2)
    A = np.eye(block_dim)
    B = np.zeros((block_dim,block_dim))
    Ccustom3_matrix = np.block([[A,B],[B, custom]])
    return np.reshape(Ccustom3_matrix, (2,)*6)

def Ccustom4_tensor(custom):
    """custom is 8x8 matrix"""
    dim = 2**4
    block_dim = int(dim/2)
    A = np.eye(block_dim)
    B = np.zeros((block_dim,block_dim))
    Ccustom4_matrix = np.block([[A,B],[B, custom]])
    return np.reshape(Ccustom4_matrix, (2,)*8)

def Ccustom5_tensor(custom):
    """custom is 16x16 matrix"""
    dim = 2**5
    block_dim = int(dim/2)
    A = np.eye(block_dim)
    B = np.zeros((block_dim,block_dim))
    Ccustom5_matrix = np.block([[A,B],[B, custom]])
    return np.reshape(Ccustom5_matrix, (2,)*10)

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

def Rndag(i,n,reg): 
    reg.psi=np.tensordot(Rndag_matrix(n),reg.psi,(1,i)) 
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

def CRndag(control, target, n, reg):
    reg.psi=np.tensordot(CRndag_tensor(n), reg.psi, ((2,3),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target)) 

def CUrand(control, target, dim, reg):
    total_dim = 2**dim
    reg.psi=np.tensordot(CUrand_tensor(dim), reg.psi, ((total_dim-2,total_dim-1),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target)) 

def Ccustom2_test(control, target, custom, reg):
    reg.psi=np.tensordot(Ccustom2_tensor(custom), reg.psi, ((2,3),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target)) 

def Ccustom2(custom, reg):
    """make sure the control qubit is always first"""
    matrix_vec = Ccustom2_tensor(custom).flatten()
    matrix_op = np.reshape(matrix_vec, (4,4))
    res_vec = np.dot(matrix_op, reg.psi.flatten())
    reg.psi = np.reshape(res_vec, (2,2))

def Ccustom3(custom, reg):
    """make sure the control qubit is always first"""
    matrix_vec = Ccustom3_tensor(custom).flatten()
    matrix_op = np.reshape(matrix_vec, (8,8))
    res_vec = np.dot(matrix_op, reg.psi.flatten())
    reg.psi = np.reshape(res_vec, (2,2,2)) 

def Ccustom4(custom, reg):
    """make sure the control qubit is always first"""
    matrix_vec = Ccustom4_tensor(custom).flatten()
    matrix_op = np.reshape(matrix_vec, (16,16))
    res_vec = np.dot(matrix_op, reg.psi.flatten())
    reg.psi = np.reshape(res_vec, (2,2,2,2)) 

def Ccustom5(custom, reg):
    """make sure the control qubit is always first"""
    matrix_vec = Ccustom5_tensor(custom).flatten()
    matrix_op = np.reshape(matrix_vec, (32,32))
    res_vec = np.dot(matrix_op, reg.psi.flatten())
    reg.psi = np.reshape(res_vec, (2,2,2,2,2)) 

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
    rho = np.outer(np.conj(moved_psi), moved_psi)
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

def randstate(flag_count, nqubits):
    """Calculate random state"""
    dim = int((2**nqubits) / (2**flag_count))
    raw = np.random.uniform(0,1,dim)
    denom = np.sqrt(np.sum(np.square(raw)))
    dict = {}
    for i in range(dim):
        bin = np.binary_repr(i)
        if i==0:
            bin = '00'
        elif i==1:
            bin = '01'
        binint = tuple([int(bit) for bit in bin])
        dict[(0,)*flag_count+binint] = raw[i]/denom
    return dict
    #setstate(D, reg)

def fidelity(rho1, rho2):
    try:
        rho1_sqrt = sqrtm(rho1)
        prod = np.matmul(rho1_sqrt,(np.matmul(rho2, rho1_sqrt)))
        return np.real(np.trace(sqrtm(prod)))
    except:
        return None
