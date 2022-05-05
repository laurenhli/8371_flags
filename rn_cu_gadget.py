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

def CU_flag(start, n, flag, q1, q2, err=False, randuflag=True):
    if randuflag: #create random unitary flag
        F = unitary_group.rvs(4)
    else: #random pauli flag
        F = Pauliflag[np.random.choice(Paulierr)]
    U = CRn_matrix(n)
    Fdag = np.array(np.matrix(F).getH())
    Udag = np.array(np.matrix(U).getH())
    Fprime = multi_dot([U, Fdag, Udag])
    
    #clean for fidelity calculation
    reg2=Reg(3)
    setstate(start, reg2)
    CRn(q1, q2, n, reg2)
    rho2 = reducedrho([flag], reg2)

    #flag circuit
    reg=Reg(3)
    setstate(start, reg)
    H(flag, reg)
    Ccustom3(F, reg) #random flag F
    CRn(q1, q2, n, reg) #two-qubit gate U

    if err: ## error here
        if err[0] == 'X':
            X(q1, reg)
        elif err[0] == 'Y':
            Y(q1, reg)
        elif err[0] == 'Z':
            Z(q1, reg)

        if err[1] == 'X':
            X(q2, reg)
        elif err[1] == 'Y':
            Y(q2, reg)
        elif err[1] == 'Z':
            Z(q2, reg)

    Ccustom3(Fprime, reg) #flag Fprime = U Fdag Udag
    H(flag, reg) 
    rho = reducedrho([flag], reg)
    return measure(flag, reg), fidelity(rho, rho2)

def flag_fid(nflags, n, flag, q1, q2, p, err=False, randuflag=True):
    fid_res = []
    i = 0
    rejected = 0
    while i < nflags:
        if np.random.uniform(0,1) < p: #error occured
            err = np.random.choice(Paulierr)
        flag_res, fid = CU_flag(D00, n, flag, q1, q2, err, randuflag)
        if flag_res == 1:
            rejected += 1
            continue
        fid_res.append(fid)
        #print('flag#', i, end='\r')
        i+=1
    return fid_res, rejected/(nflags+rejected) #reject rate




