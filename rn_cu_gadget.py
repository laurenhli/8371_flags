import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import norm
import matplotlib.pyplot as plt
from helper import *

D00 = {(0,0,0):1}
D01 = {(0,0,0):0, (0,0,1):1}
D10 = {(0,0,0):0, (0,1,0):1}
D11 = {(0,0,0):0, (0,1,1):1}
even_start = {(0,0,0):0.5,
              (0,0,1):0.5,
              (0,1,0):0.5,
              (0,1,1):0.5}
Paulierr = ['IX', 'IY', 'IZ',
   'XI', 'XX', 'XY', 'XZ',
   'YI', 'YX', 'YY', 'YZ',
   'ZI', 'ZX', 'ZY', 'ZZ']
Pauliflag = {'IX':np.kron(I_matrix, X_matrix),
             'IY':np.kron(I_matrix, Y_matrix),
             'IZ':np.kron(I_matrix, Z_matrix),
             'XI':np.kron(X_matrix, I_matrix),
             'XX':np.kron(X_matrix, X_matrix),
             'XY':np.kron(X_matrix, Y_matrix),
             'XZ':np.kron(X_matrix, Z_matrix),
             'YI':np.kron(Y_matrix, I_matrix),
             'YX':np.kron(Y_matrix, X_matrix),
             'YY':np.kron(Y_matrix, Y_matrix),
             'YZ':np.kron(Y_matrix, Z_matrix),
             'ZI':np.kron(Z_matrix, I_matrix),
             'ZX':np.kron(Z_matrix, X_matrix),
             'ZY':np.kron(Z_matrix, Y_matrix),
             'ZZ':np.kron(Z_matrix, Z_matrix)}
onePaulierr = {'I':I_matrix,'X':X_matrix,'Y':Y_matrix,'Z':Z_matrix,}

def apply_noise(i,j,p, reg, err=False):
    if err==False:
        if np.random.uniform(0,1) < p: #error occured
            err = np.random.choice(Paulierr)
    if err != False:
        ## error here
        if err[0] == 'X':
            X(i, reg)
        elif err[0] == 'Y':
            Y(i, reg)
        elif err[0] == 'Z':
            Z(i, reg)

        if err[1] == 'X':
            X(j, reg)
        elif err[1] == 'Y':
            Y(j, reg)
        elif err[1] == 'Z':
            Z(j, reg)

def CU_flag(state_dict, n, flag, q1, q2, p, randuflag=True, err=False):
    if randuflag: #create random unitary flag
        F = unitary_group.rvs(4)
    else: #random pauli flag
        F = Pauliflag[np.random.choice(Paulierr)]
    U = CRn_matrix(n)
    Fdag = np.array(np.matrix(F).getH())
    Udag = np.array(np.matrix(U).getH())
    Fprime = multi_dot([U, Fdag, Udag])

    #random or even initial state
    #state_dict = randstate(1, 3)
    #state_dict = even_start
    
    #clean for fidelity calculation
    reg2=Reg(3)
    setstate(state_dict, reg2)
    CRn(q1, q2, n, reg2)
    rho2 = reducedrho([flag], reg2)

    #flag circuit
    reg=Reg(3)
    setstate(state_dict, reg)
    H(flag, reg)
    Ccustom3(F, reg) #random flag F
    CRn(q1, q2, n, reg) #two-qubit gate U
    apply_noise(q1, q2, p, reg, err)
    
    Ccustom3(Fprime, reg) #flag Fprime = U Fdag Udag
    H(flag, reg) 
    rho = reducedrho([flag], reg)
    fid_calc = fidelity(rho, rho2)
    if fid_calc != None:
        return measure(flag, reg), fid_calc
    else:
        return 'failed', 'failed'

def flag_fid(state_dict, nflags, n, flag, q1, q2, p, randuflag=True, err=False):
    fid_res = []
    fid_rej = []
    rejected = 0
    for _ in range(nflags):
        flag_res, fid = CU_flag(state_dict, n, flag, q1, q2, p, randuflag, err)
        if flag_res == 'failed':
            continue

        if flag_res == 1:
            rejected += 1
            fid_rej.append(fid)
        else:
            fid_res.append(fid)
        #print('flag#', i, end='\r')

    return fid_res, fid_rej, rejected/nflags #reject rate




