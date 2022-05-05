import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import norm
import matplotlib.pyplot as plt
from rn_cu_gadget import *
from helper import *

D00 = {(0,0,0):1}

#r2 matrix (4x4 matrix)
r2 = CRn_matrix(2)

#r3 matrix (8x8 matrix)
r3_A = np.eye(4)
r3_B = np.zeros((4,4))
r3_C = np.kron(I_matrix, Rn_matrix(3))
r3 = np.block([[r3_A, r3_B],[r3_B, r3_C]])

#r4 matrix (16x16 matrix)
r4_A = np.eye(8)
r4_B = np.zeros((8,8))
r4_C = np.kron(np.eye(4), Rn_matrix(4))
r4 = np.block([[r4_A, r4_B],[r4_B, r4_C]])

#4x4 matrix
s1 = np.kron(I_matrix, H_matrix)
s2 = r2
s3 = np.kron(H_matrix, I_matrix)
qft2 = multi_dot([s3,s2,s1])

#8x8 matrix
u1 = np.kron(np.eye(4), H_matrix)
u2 = np.kron(I_matrix, r2)
u3 = r3
u4 = np.kron(I_matrix, np.kron(H_matrix, I_matrix))
u5 = np.kron(r2, I_matrix)
u6 = np.kron(H_matrix, np.eye(4))
qft3 = multi_dot([u6, u5, u4, u3, u2, u1])

#16x16 matrix
v1 = np.kron(np.eye(8), H_matrix)
v2 = np.kron(np.eye(4), r2)
v3 = np.kron(I_matrix, r3)
v4 = r4
v5 = np.kron(np.eye(4), np.kron(H_matrix, I_matrix))
v6 = np.kron(I_matrix, np.kron(r2, I_matrix))
v7 = np.kron(r3, I_matrix)
v8 = np.kron(I_matrix, np.kron(H_matrix, np.eye(4)))
v9 = np.kron(r2, np.eye(4))
v10 = np.kron(H_matrix, np.eye(8))
qft4 = multi_dot([v10, v9, v8, v7, v6, v5, v4, v3, v2, v1])

def qft2_circ(qubits, reg, p):
    q1 = qubits[0]
    q2 = qubits[1]
    H(q2, reg)
    CRn(q1, q2, 2, reg)
    apply_noise(q1, q2, p, reg)
    H(q1, reg)

def qft3_circ(qubits, reg, p):
    q1 = qubits[0]
    q2 = qubits[1]
    q3 = qubits[2]
    H(q3, reg)
    CRn(q2, q3, 2, reg)
    apply_noise(q2, q3, p, reg)
    CRn(q1, q3, 3, reg)
    apply_noise(q1, q3, p, reg)
    H(q2, reg)
    CRn(q1, q2, 2, reg)
    apply_noise(q1, q2, p, reg)
    H(q1, reg)

def qft4_circ(qubits, reg, p):
    q1 = qubits[0]
    q2 = qubits[1]
    q3 = qubits[2]
    q4 = qubits[3]
    H(q4, reg)
    CRn(q3, q4, 2, reg)
    apply_noise(q3, q4, p, reg)
    CRn(q2, q4, 3, reg)
    apply_noise(q2, q4, p, reg)
    CRn(q1, q4, 4, reg)
    apply_noise(q1, q4, p, reg)

    H(q3, reg)
    CRn(q2, q3, 2, reg)
    apply_noise(q2, q3, p, reg)
    CRn(q1, q3, 3, reg)
    apply_noise(q1, q3, p, reg)
    
    H(q2, reg)
    CRn(q1, q2, 2, reg)
    apply_noise(q1, q2, p, reg)

    H(q1, reg)

def add_flag(dim, F_matrix, reg):
    if dim == 2:
        Ccustom2(F_matrix, reg)
    elif dim == 3:
        Ccustom3(F_matrix, reg)
    elif dim == 4:
        Ccustom4(F_matrix, reg)
    elif dim == 5:
        Ccustom5(F_matrix, reg)

def get_U_matrix(dim):
    if dim == 2:
        return qft2
    elif dim == 3:
        return qft3
    elif dim == 4:
        return qft4

def get_pauli_F(dim):
    if dim == 2:
        return Pauliflag[np.random.choice(Paulierr)]
    if dim > 2:
        err = np.random.choice(['I','X','Y','Z'], dim)
        flag = np.kron(onePaulierr[err[1]], onePaulierr[err[0]])
        for i in range(2,dim):
            flag = np.kron(onePaulierr[err[i]],flag)
        return flag

def qft_flag_circ(state_dict, flag, qubits, qft_circ, p, randuflag=True):
    nqubits = len(qubits)
    qb_dim = 2**nqubits
    if randuflag: #create random unitary flag
        F = unitary_group.rvs(qb_dim)
    else: #random pauli flag
        F = get_pauli_F(nqubits)
        #F = Pauliflag[np.random.choice(Paulierr)]
    U = get_U_matrix(nqubits)
    Fdag = np.array(np.matrix(F).getH())
    Udag = np.array(np.matrix(U).getH())
    Fprime = multi_dot([U, Fdag, Udag])

    #clean for fidelity calculation
    reg2=Reg(nqubits+1)
    setstate(state_dict, reg2)
    qft_circ(qubits, reg2, 0)
    rho2 = reducedrho([flag], reg2)

    #flag circuit
    reg=Reg(nqubits+1)
    setstate(state_dict, reg)
    H(flag, reg)
    add_flag(nqubits+1, F, reg) #random flag F
    qft_circ(qubits, reg, p)
    add_flag(nqubits+1, Fprime, reg) #flag Fprime = U Fdag Udag
    H(flag, reg) 

    rho = reducedrho([flag], reg)
    fid_calc = fidelity(rho, rho2)
    if fid_calc != None:
        return measure(flag, reg), fid_calc
    else:
        return 'failed', 'failed'

def qft_fid(state_dict, nflags, flag, qubits, qft_circ, p, randuflag=True):
    fid_res = []
    fid_rej = []
    rejected = 0
    for _ in range(nflags):
        flag_res, fid = qft_flag_circ(state_dict, flag, qubits, qft_circ, p, randuflag)
        if flag_res == 'failed':
            continue

        if flag_res == 1:
            rejected += 1
            fid_rej.append(fid)
        else:
            fid_res.append(fid)
        #print('flag#', i, end='\r')

    return fid_res, fid_rej, rejected/nflags #reject rate

def qft_plot_data(state_dict, nflags, flag, qubits, qft_circ, probs, randuflag=False):
    rej_prop_all = []
    #res_all = []
    #rej_all = []

    res_temp = []
    rej_temp = []
    x_res = []
    x_rej = []
    for p in probs:
        print(p,end='\r')
        res, rej, rej_prop = qft_fid(state_dict, nflags, flag, qubits, qft_circ, p, randuflag)
        rej_prop_all.append(rej_prop)
        res_temp += res
        rej_temp += rej
        x_res += [p]*len(res)
        x_rej += [p]*len(rej)
        #res_all.append(np.average(res))
        #rej_all.append(np.average(rej))
    
    return [x_res, res_temp], [x_rej, rej_temp], rej_prop_all#, res_all, rej_all