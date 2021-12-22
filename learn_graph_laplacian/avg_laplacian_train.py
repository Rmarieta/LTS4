import numpy as np
from cvxopt import matrix, solvers
import networkx as nx
from tqdm import tqdm
import argparse
import os, shutil
import collections
import dill as pickle
import progressbar
import matplotlib.pyplot as plt
import utils
import time

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def build_dir(input, sets, types, output) :
    
    if not os.path.exists(input) : exit('Input data directory does not exist')

    for set in sets :
        for type in types :
            if not os.path.exists(os.path.join(input,set,type)) :
                exit('Extracted data for the selected class (%s) is not available'%type)

    # Check if the output folder exists
    if not os.path.exists(output) :
        os.makedirs(output)
    else : # If it already exists, empty the subfolders
        for dir_file in os.listdir(output) :
            file = os.path.join(output, dir_file)
            try:
                if os.path.isfile(file) or os.path.islink(file):
                    os.unlink(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file, e))
    
    # Create one output folder per set (dev or train)
    for set in sets :
        for type in seizure_types :
            szr_dir = os.path.join(output,set,type)
            if not os.path.exists(szr_dir) :
                os.makedirs(szr_dir)
    
def load_pickle(file_path) :
    data = pickle.load(open(file_path, 'rb'))
    time_series_data = data.data
    return np.array(time_series_data)

def gl_sig_model(inp_signal, max_iter, alpha, beta, ignore_L):
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
        if not ignore_L : assert np.all(L - np.diag(np.diag(L)) <= 0)
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

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def compute_Laplacian(input_dir, set, types, restrict_size, ignore_L) :

    # np.random.seed(0)
    solvers.options['show_progress'] = False
    chop_size = 40000

    # Create Laplacian dictionary and fill it up with the learned L's
    Laplacian_dict = collections.defaultdict(list)

    for szr_type in types :
        type_dir = os.path.join(input_dir,set,szr_type)

        for root, dir, files in os.walk(type_dir) :
            
            print('\nComputation of the graphs for',set+'/'+szr_type,'in progress...\n')
            bar = progressbar.ProgressBar(maxval=len(files),
                                        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            count = 0

            for file in files[:2] :
                
                patient_id = file.split('_')[3]
                
                input = np.transpose(load_pickle(os.path.join(type_dir,file))) # Extract the [nx20] array of the .pkl file
                
                if (np.amax(input) - np.amin(input)) != 0 : # In some instances, the input signal is 0 for all t, we discard these samples

                    #input = (input-np.amin(input))/(np.amax(input)-np.amin(input)) # Normalize the input
                    print('Max value : ',np.amax(np.abs(input)))

                """
                    if restrict_size :
                        chop_idx = [0, chop_size]
                    else :
                        nb_sections = len(input)//chop_size+1 # Integer division
                        # Define even sections to avoid having a graph computed from short section which can induce a bias in the averaged matrix
                        chop_idx = np.linspace(0,len(input),num=nb_sections+1).astype(int)
                    
                    A_list = []

                    for i in range(len(chop_idx)-1) :

                        L, _ = gl_sig_model(inp_signal=input[chop_idx[i]:chop_idx[i+1]], max_iter=1, alpha=1, beta=5, ignore_L=ignore_L)

                        # To get the adjacency matrix
                        A_tmp = -(L - np.diag(np.diag(L)))
                        
                        # Add to list for future average computation
                        A_list.append(A_tmp)
                    

                    A = sum(A_list)/len(A_list) # Average the adjacency matrices                    
                    A = A/np.amax(A.flatten()) # Normalise A
                    
                    # Print the adjacency matrix
                    # print('A :\n',np.around(A[:10,:10],decimals=3))
                    
                    Laplacian_dict[szr_type].append((patient_id, A))
                
                """
                count +=1
                bar.update(count)

            bar.finish()

    # To get the graph back from the adjacency matrix :
    # G = nx.from_numpy_matrix(W)

    return Laplacian_dict

def save_graphs(dict, output, set) : 
    
    L_dir = os.path.join(output,set)

    for seizure_type, Laplacians in dict.items():
        count = 0
        for L in Laplacians : # L is a tuple of the form (patient_id, adjacency_matrix)
            np.save(os.path.join(L_dir,seizure_type,"graph_" + str(count) + "_pid_" + L[0] + "_" + seizure_type + ".npy"), L[1])
            count += 1
        
        print(set+'/'+seizure_type+' done')

if __name__ == "__main__":

    print('\n\nSTART\n\n')

    parser = argparse.ArgumentParser(description='Learn the graph laplacian matrix from the EEG samples')
    parser.add_argument('--data_dir', default='./data', help='path to the dataset')
    known_args, _ = parser.parse_known_args()
    data_dir = known_args.data_dir

    parser.add_argument('--input_dir', default=os.path.join(data_dir,'v1.5.2/raw_samples'), help='path to the input for the Laplacian computation')
    parser.add_argument('--seizure_types',default=['BG','FNSZ','GNSZ'], help="types of seizures for which we compute the Laplacian (include 'BG' if needed), in the form --seizure_types 'BG' 'FNSZ' 'GNSZ'", nargs="+")
    parser.add_argument('--graph_dir', default=os.path.join(data_dir,'v1.5.2/graph_output'), help='path to the output of the Laplacian computation')
    parser.add_argument('--restrict_size', default=True, help='restrict the size of the EEG recordings to avoid crashing', type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--ignore_L', default=True, help='ignore the assert in the Laplacian computation', type=lambda x: (str(x).lower() in ['true','1']))

    args = parser.parse_args()
    seizure_types = args.seizure_types
    input_dir = args.input_dir
    graph_dir = args.graph_dir
    restrict_size = args.restrict_size
    ignore_L = args.ignore_L

    # Check that the appropriate folders already exist and build/empty the output folders
    build_dir(input_dir, ['train'], seizure_types, graph_dir)
    
    # Same for the train dataset
    Laplacian_train_dict = compute_Laplacian(input_dir, 'train', seizure_types, restrict_size, ignore_L)  
    print('\n\nSaving the graphs as .npy files...\n')
    #save_graphs(Laplacian_train_dict, graph_dir, 'train')
    print('\n...Saving done\n\n')
    
    print('\n\nDONE\n\n')

    

    