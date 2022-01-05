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

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def compute_cov(input_dir, set, types) :

    # np.random.seed(0)
    solvers.options['show_progress'] = False
    chop_size = 000

    # Create matrix dictionary and fill it up with the learned C's
    cov_dict = collections.defaultdict(list)

    for szr_type in types :
        type_dir = os.path.join(input_dir,set,szr_type)

        for root, dir, files in os.walk(type_dir) :
            
            print('\nComputation of the graphs for',set+'/'+szr_type,'in progress...\n')
            bar = progressbar.ProgressBar(maxval=len(files),
                                        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            count = 0

            for file in files :
                
                patient_id = file.split('_')[3]
                
                input = load_pickle(os.path.join(type_dir,file)) # Extract the [nx20] array of the .pkl file
                
                if (np.amax(input) - np.amin(input)) != 0 : # In some instances, the input signal is 0 for all t, we discard these samples

                    input = input/np.amax(np.abs(input)) # Normalize the input

                    # Get the covariance matrix and ReLU
                    C = np.maximum(np.cov(input),0)              
                    
                    cov_dict[szr_type].append((patient_id, C))

                count +=1
                bar.update(count)

            bar.finish()

    # To get the graph back from the adjacency matrix :
    # G = nx.from_numpy_matrix(W)

    return cov_dict

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

    args = parser.parse_args()
    seizure_types = args.seizure_types
    input_dir = args.input_dir
    graph_dir = args.graph_dir

    # Check that the appropriate folders already exist and build/empty the output folders
    build_dir(input_dir, ['dev','train'], seizure_types, graph_dir)
    
    # For the dev dataset
    cov_dev_dict = compute_cov(input_dir, 'dev', seizure_types) 
    print('\n\nSaving the graphs as .npy files...\n')
    save_graphs(cov_dev_dict, graph_dir, 'dev')
    print('\n...Saving done\n\n')

    # Same for the train dataset
    cov_train_dict = compute_cov(input_dir, 'train', seizure_types) 
    print('\n\nSaving the graphs as .npy files...\n')
    save_graphs(cov_train_dict, graph_dir, 'train')
    print('\n...Saving done\n\n')
    
    print('\n\nDONE\n\n')

    

    