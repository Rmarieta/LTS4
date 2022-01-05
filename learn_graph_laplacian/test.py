from matplotlib.colors import Normalize
import numpy as np
from cvxopt import matrix, solvers
import networkx as nx
from numpy.linalg import norm
from tqdm import tqdm
import argparse
import os, shutil
import collections
import dill as pickle
import progressbar
import matplotlib.pyplot as plt

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def load_pickle(file_path) :
    data = pickle.load(open(file_path, 'rb'))
    time_series_data = data.data
    return np.array(time_series_data)

def gl_sig_model(inp_signal, max_iter, alpha, beta):
    """
    Returns Output Signal Y, Graph Laplacian L
    """
    Y = inp_signal.T
    num_vertices = inp_signal.shape[1]
    M_mat, P_mat, A_mat, b_mat, G_mat, h_mat = create_static_matrices_for_L_opt(num_vertices, beta)
    # M_c = matrix(M_mat)
    P_c = matrix(P_mat)
    A_c = matrix(A_mat)
    b_c = matrix(b_mat)
    G_c = matrix(G_mat)
    h_c = matrix(h_mat)
    curr_cost = np.linalg.norm(np.ones((num_vertices, num_vertices)), 'fro')
    q_mat = alpha * np.dot(np.ravel(np.dot(Y, Y.T)), M_mat)
    for it in range(max_iter):
        # pdb.set_trace()
        # Update L
        prev_cost = curr_cost
        # pdb.set_trace()
        q_c = matrix(q_mat)
        sol = solvers.qp(P_c, q_c, G_c, h_c, A_c, b_c)
        l_vech = np.array(sol['x'])
        l_vec = np.dot(M_mat, l_vech)
        L = l_vec.reshape(num_vertices, num_vertices)
        # Assert L is correctly learnt.
        # assert L.trace() == num_vertices
        assert np.allclose(L.trace(), num_vertices)
        assert np.all(L - np.diag(np.diag(L)) <= 0)
        assert np.allclose(np.dot(L, np.ones(num_vertices)), np.zeros(num_vertices))
        # print('All constraints satisfied')
        # Update Y
        Y = np.dot(np.linalg.inv(np.eye(num_vertices) + alpha * L), inp_signal.T)

        curr_cost = (np.linalg.norm(inp_signal.T - Y, 'fro')**2 +
                     alpha * np.dot(np.dot(Y.T, L), Y).trace() +
                     beta * np.linalg.norm(L, 'fro')**2)
        q_mat = alpha * np.dot(np.ravel(np.dot(Y, Y.T)), M_mat)
        # pdb.set_trace()
        calc_cost = (0.5 * np.dot(np.dot(l_vech.T, P_mat), l_vech).squeeze() +
                     np.dot(q_mat, l_vech).squeeze() + np.linalg.norm(inp_signal.T - Y, 'fro')**2)
        # pdb.set_trace()
        assert np.allclose(curr_cost, calc_cost)
        # print(curr_cost)
        if np.abs(curr_cost - prev_cost) < 1e-4:
            # print('Stopped at Iteration', it)
            break
        # print
    return L, Y

def create_static_matrices_for_L_opt(num_vertices, beta):
    # Static matrices are those independent of Y
    #
    M_mat = create_dup_matrix(num_vertices)
    P_mat = 2 * beta * np.dot(M_mat.T, M_mat)
    A_mat = create_A_mat(num_vertices)
    b_mat = create_b_mat(num_vertices)
    G_mat = create_G_mat(num_vertices)
    h_mat = np.zeros(G_mat.shape[0])
    return M_mat, P_mat, A_mat, b_mat, G_mat, h_mat

def get_u_vec(i, j, n):
    u_vec = np.zeros(n*(n+1)//2)
    pos = (j-1) * n + i - j*(j-1)//2
    u_vec[pos-1] = 1
    return u_vec

def get_T_mat(i, j, n):
    Tij_mat = np.zeros((n, n))
    Tij_mat[i-1, j-1] = Tij_mat[j-1, i-1] = 1
    return np.ravel(Tij_mat)

def create_dup_matrix(num_vertices):
    M_mat = np.zeros((num_vertices**2, num_vertices*(num_vertices + 1)//2))
    # tmp_mat = np.arange(num_vertices**2).reshape(num_vertices, num_vertices)
    for j in range(1, num_vertices+1):
        for i in range(j, num_vertices+1):
            u_vec = get_u_vec(i, j, num_vertices)
            Tij = get_T_mat(i, j, num_vertices)
            # pdb.set_trace()
            M_mat += np.outer(u_vec, Tij).T

    return M_mat

def get_a_vec(i, n):
    a_vec = np.zeros(n*(n+1)//2)
    if i == 0:
        a_vec[np.arange(n)] = 1
    else:
        tmp_vec = np.arange(n-1, n-i-1, -1)
        tmp2_vec = np.append([i], tmp_vec)
        tmp3_vec = np.cumsum(tmp2_vec)
        a_vec[tmp3_vec] = 1
        end_pt = tmp3_vec[-1]
        a_vec[np.arange(end_pt, end_pt + n-i)] = 1

    return a_vec

def create_A_mat(n):
    A_mat = np.zeros((n+1, n*(n+1)//2))
    # A_mat[0, 0] = 1
    # A_mat[0, np.cumsum(np.arange(n, 0, -1))] = 1
    for i in range(0, A_mat.shape[0] - 1):
        A_mat[i, :] = get_a_vec(i, n)
    A_mat[n, 0] = 1
    A_mat[n, np.cumsum(np.arange(n, 1, -1))] = 1

    return A_mat

def create_b_mat(n):
    b_mat = np.zeros(n+1)
    b_mat[n] = n
    return b_mat

def create_G_mat(n):
    G_mat = np.zeros((n*(n-1)//2, n*(n+1)//2))
    tmp_vec = np.cumsum(np.arange(n, 1, -1))
    tmp2_vec = np.append([0], tmp_vec)
    tmp3_vec = np.delete(np.arange(n*(n+1)//2), tmp2_vec)
    for i in range(G_mat.shape[0]):
        G_mat[i, tmp3_vec[i]] = 1

    return G_mat

def get_precision_er(w_out, w_gt):
    num_cor = 0
    tot_num = 0
    for r in range(w_out.shape[0]):
        for c in range(w_out.shape[1]):
            if w_out[r, c] > 0:
                tot_num += 1
                if w_gt[r, c] > 0:
                    num_cor += 1
    # print(num_cor, tot_num, num_cor / tot_num)
    return num_cor / tot_num

def get_precision_er_L(L_out, L_gt, thresh=1e-4):
    W_out = -L_out
    np.fill_diagonal(W_out, 0)
    W_out[W_out < thresh] = 0
    # pdb.set_trace()
    W_gt = -L_gt.todense()
    np.fill_diagonal(W_gt, 0)
    return get_precision_er(W_out, W_gt)

def get_recall_er_L(L_out, L_gt, thresh=1e-4):
    W_out = -L_out
    np.fill_diagonal(W_out, 0)
    W_out[W_out < thresh] = 0
    # pdb.set_trace()
    W_gt = -L_gt.todense()
    np.fill_diagonal(W_gt, 0)
    return get_precision_er(W_gt, W_out)

def get_precision_rnd(w_out, w_gt):
    num_cor = 0
    tot_num = 0
    for r in range(w_out.shape[0]):
        for c in range(w_out.shape[1]):
            if w_out[r, c] > 0:
                tot_num += 1
                if w_gt[r, c] > 0:
                    num_cor += 1
    # print(num_cor, tot_num, num_cor / tot_num)
    return num_cor / tot_num

def get_prec_recall_rnd_L(L_out, L_gt, thresh=1e-4):
    W_out = -L_out
    np.fill_diagonal(W_out, 0)
    W_out[W_out < thresh] = 0
    # pdb.set_trace()
    W_gt = -L_gt.todense()
    np.fill_diagonal(W_gt, 0)
    prec = get_precision_rnd(W_out, W_gt)
    rec = get_precision_rnd(W_gt, W_out)
    return prec, rec

def get_f_score(prec, recall):
    return 2 * prec * recall / (prec + recall)

def get_MSE(L_out, L_gt):
    return np.linalg.norm(L_out - L_gt, 'fro')


if __name__ == "__main__":

    seizure_types = ['BG','FNSZ','GNSZ']

    # for i = 2 => crashes with FNSZ ("./data/v1.5.2/raw_samples/dev/FNSZ/file_2_pid_00006546_type_FNSZ.pkl")

    files = ["./data/v1.5.2/raw_samples/dev/FNSZ/file_4_pid_00006546_type_FNSZ.pkl",
             "./data/v1.5.2/raw_samples/dev/BG/file_4_pid_00000258_type_BG.pkl",
             "./data/v1.5.2/raw_samples/dev/GNSZ/file_4_pid_00004671_type_GNSZ.pkl"]



    if False :

        for file in files :
            plt.figure()
            input = load_pickle(file) # Extract the [nx20] array of the .pkl file
            print('\nShape input : ',input.shape,'\n')

            ## input = DCT(input) => of same length as input (discrete cosine transform), then keep first 20% of signal for ex. (or after 3000 time samples)

            # Then A = -(L - diag(L)) (= adjacency matrix, which is the graph we are looking for)
            # Normalise the A's => A = A/np.linalg.norm(A,'fro')
            # Do the mean in the A if there's long samples
            
            if False :
                input = (input/np.amax(input))[:15000]

            #L, Y = gl_sig_model(inp_signal=input, max_iter=1, alpha=.2, beta=5) # Higher beta = more connexions
            
            L = np.cov(input)
            print('\nShape of L : ',L.shape,'\n')
            print('Sample :\n',np.maximum(np.around(L[:7,:7],2),0),'\n=> symmetric !\n')
            
            A = -(L - np.diag(np.diag(L)))
            A = A/np.amax(A.flatten())
            #A = A/np.linalg.norm(A,'fro')
            plt.imshow(A, cmap='gray')

            # A is symmetrical => only keep half (without diagonal)
            
            # TRY ON GOOGLE COLLAB WITH THE SAMPLES with shape[0] > 28000

            plt.colorbar()

    else :

        files = ['./data/v1.5.2/graph_cov/dev/GNSZ/graph_0_pid_00010062_GNSZ.npy',
        './data/v1.5.2/graph_cov/dev/GNSZ/graph_1_pid_00008479_GNSZ.npy',
        './data/v1.5.2/graph_cov/dev/GNSZ/graph_2_pid_00006546_GNSZ.npy']
        for F in files :
                
            graph = np.load(F)
            print('\n',np.around(graph[:10,:10],decimals=3),'\n')

    plt.show()
    