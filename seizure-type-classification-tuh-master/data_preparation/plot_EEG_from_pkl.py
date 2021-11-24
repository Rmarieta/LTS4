import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import dill as pickle
import collections
import numpy as np
import warnings
import matplotlib.pyplot as plt

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def plot_pickle(file_path, plot):

    warnings.filterwarnings("ignore")
    type_data = pickle.load(open(file_path, 'rb'))
    time_series_data = type_data.data

    time_series_data = np.array(time_series_data)
    
    print('Length of seizure crop : ',time_series_data.shape[1],'\n')
    
    if plot :
    
        x1 = range(len(time_series_data[0,:]))

        L = range(len(time_series_data[:,0]))

        fig, axs = plt.subplots(len(L))

        for i in L :
            axs[i].plot(x1,time_series_data[i,:])
            axs[i].axis('off')

plt.show()

def main():

    print('Plot the EEG time series from the pickle files')

    save_data_dir = '../data'
    preprocess_data_dir = '../data'

    tuh_eeg_szr_ver = 'v1.5.2'

    save_data_dir = os.path.join(save_data_dir,tuh_eeg_szr_ver,'raw_seizures')
    preprocess_data_dir = os.path.join(preprocess_data_dir,tuh_eeg_szr_ver,'output','fft')

    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(save_data_dir):
        fnames.extend(filenames)

    nb_file = 30

    paths = [os.path.join(save_data_dir,f) for f in fnames[:nb_file]]
    print('\nPath : ',paths,'\n')

    for file_path in paths:
        plot_pickle(file_path,plot=False)
        print('File done')
    
    plt.show()


if __name__ == '__main__':
    main()




