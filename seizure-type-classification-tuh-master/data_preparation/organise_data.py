import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse
import pandas as pd
import numpy as np
import math
import collections
from tabulate import tabulate
import pyedflib
import re
from scipy.signal import resample
import pickle
import h5py
import progressbar
from time import sleep

# Run the following in \seizure-type-classification-tuh-master
# python data_preparation/organise_data.py --base_dir ../data --tuh_eeg_szr_ver v1.5.2 -- merge_train_dev False
# Last argument not needed : python data_preparation/organise_data.py --base_dir ../data

""" 
This file first generates the data dict out of the excel file (for both dev and train sets using the multiple sheets
from the excel file), then merge the one from dev and train, and finally loads all the .edf files of the dataset 
into pickle files (that will be located in data/v1.5.2/raw_seizures)
"""

parameters = pd.read_csv('data_preparation/parameters.csv', index_col=['parameter'])
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def generate_data_dict(xlsx_file_name, sheet_name, tuh_eeg_szr_ver):

    seizure_info = collections.namedtuple('seizure_info', ['patient_id','filename', 'start_time', 'end_time'])
    data_dict = collections.defaultdict(list)

    excel_file = os.path.join(xlsx_file_name)
    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    if tuh_eeg_szr_ver == 'v1.5.2':
        data = data.iloc[1:] # remove first row
    else:
        exit('tuh_eeg_szr_ver %s is not supported'%tuh_eeg_szr_ver)

    col_l_file_name = data.columns[11]
    col_m_start = data.columns[12]
    col_n_stop = data.columns[13]
    col_o_szr_type = data.columns[14]
    train_files = data[[col_l_file_name, col_m_start,col_n_stop,col_o_szr_type]]
    
    print(train_files)

    """
    Excel sheet explanation :
    Every time there's a change in the last folder (e.g. s007_2012_03_15) then the number of seizures recorded in 
    that folder (there might be multiple .edf files) is displayed only on the first line.

    Options :
    1) Retrieve all samples from the patients that have FSZ or GSZ seizures, even the full recordings when no seizure
    was recorded 
    => retrieve the folder name (before-last), e.g. s007_2012_03_15, and retrieve all the filenames inside the folders,
    => get all the segments with no seizures
    2) Simply keep the samples with a seizure in the recording (very easy)
    => when the samples are cut using start/end time i.e. EEG[t_0:t_1], store EEG[0:t_0] and EEG[t_1:] as background
    crops (watch out if there's more than one seizure in the recording)
    """
    
    """
    GOAL :
    Not remove all lines that are normal but rather extract all seizures the same way and put them into GSZ or FSZ,
    and then keep the normal seizure lines => set start_time/end_time to None or to t_0, t_end depending on how we
    want to handle the case of a normal .edf file extraction

    To select only the general seizures, for example we could drop all train_files that do not have 'GNSZ' in their file_name
    Ex : GNSZ_train_files = data[[col_l_file_name, col_m_start,col_n_stop,col_o_szr_type] if col_o_szr_type=='GNSZ']

    OR 
    Input : list of ['GNSZ','FNSZ', ...] seizure types
    Output : list of dict [dict('GNSZ'), dict('FNSZ'), ...]


    Then : when extracting the .edf files, create 3 folders NSZ, FSZ and GSZ (by returning 3 dict in generate_data_dict
    and running the extraction 3 times ?)
    """


    train_files = np.array(train_files.dropna())

    print(train_files)
    if True : 
        exit('BLABLABLA')

    for item in train_files:
        a = item[0].split('/')
        if tuh_eeg_szr_ver == 'v1.5.2':
            patient_id = a[4]
        else:
            exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

        v = seizure_info(patient_id = patient_id, filename = item[0], start_time=item[1], end_time=item[2])
        k = item[3] # szr_type
        data_dict[k].append(v)

    return data_dict

def print_type_information(data_dict):

    l = []
    for szr_type, szr_info_list in data_dict.items():
        # how many different patient id for seizure K?
        patient_id_list = [szr_info.patient_id for szr_info in szr_info_list]
        unique_patient_id_list,counts = np.unique(patient_id_list,return_counts=True)

        dur_list = [szr_info.end_time-szr_info.start_time for szr_info in szr_info_list]
        total_dur = sum(dur_list)
        # l.append([szr_type, str(len(szr_info_list)), str(len(unique_patient_id_list)), str(total_dur)])
        l.append([szr_type, (len(szr_info_list)), (len(unique_patient_id_list)), (total_dur)])

        #  numpy.asarray((unique, counts)).T
        '''
        if szr_type=='TNSZ':
            print('TNSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        if szr_type=='SPSZ':
            print('SPSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        '''

    sorted_by_szr_num = sorted(l, key=lambda tup: tup[1], reverse=True)
    print(tabulate(sorted_by_szr_num, headers=['Seizure Type', 'Seizure Num','Patient Num','Duration(Sec)']))

def merge_train_test(train_data_dict, dev_test_data_dict):

    merged_dict = collections.defaultdict(list)
    for item in train_data_dict:
        merged_dict[item] = train_data_dict[item] + dev_test_data_dict[item]

    return merged_dict

def extract_signal(f, signal_labels, electrode_name, start, stop):

    tuh_label = [s for s in signal_labels if 'EEG ' + electrode_name + '-' in s]

    if len(tuh_label) > 1:
        print(tuh_label)
        exit('Multiple electrodes found with the same string! Abort')

    channel = signal_labels.index(tuh_label[0])
    signal = np.array(f.readSignal(channel))

    start, stop = float(start), float(stop)
    original_sample_frequency = f.getSampleFrequency(channel)
    original_start_index = int(np.floor(start * float(original_sample_frequency)))
    original_stop_index = int(np.floor(stop * float(original_sample_frequency)))

    seizure_signal = signal[original_start_index:original_stop_index]

    new_sample_frequency = int(parameters.loc['sampling_frequency']['value'])
    new_num_time_points = int(np.floor((stop - start) * new_sample_frequency))
    seizure_signal_resampled = resample(seizure_signal, new_num_time_points)

    return seizure_signal_resampled

def read_edfs_and_extract(edf_path, edf_start, edf_stop):

    f = pyedflib.EdfReader(edf_path)

    montage = str(parameters.loc['montage']['value'])
    montage_list = re.split(';', montage)
    signal_labels = f.getSignalLabels()
    x_data = []

    for i in montage_list:
        electrode_list = re.split('-', i)
        electrode_1 = electrode_list[0]
        extracted_signal_from_electrode_1 = extract_signal(f, signal_labels, electrode_name=electrode_1, start=edf_start, stop=edf_stop)
        electrode_2 = electrode_list[1]
        extracted_signal_from_electrode_2 = extract_signal(f, signal_labels, electrode_name=electrode_2, start=edf_start, stop=edf_stop)
        this_differential_output = extracted_signal_from_electrode_1-extracted_signal_from_electrode_2
        x_data.append(this_differential_output)

    f._close()
    del f

    x_data = np.array(x_data)

    return x_data

def load_edf_extract_seizures_v152(base_dir, save_data_dir, data_dict):

    seizure_data_dict = collections.defaultdict(list)

    count = 0
    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for seizure_type, seizures in data_dict.items():
        for seizure in seizures:
            rel_file_location = seizure.filename.replace('.tse', '.edf').replace('./', 'edf/')
            patient_id = seizure.patient_id
            abs_file_location = os.path.join(base_dir, rel_file_location)
            try:
                temp = seizure_type_data(patient_id = patient_id, seizure_type = seizure_type, data = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time))
                with open(os.path.join(save_data_dir, 'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.pkl'), 'wb') as fseiz:
                    pickle.dump(temp, fseiz)
                count += 1
            except Exception as e:
                print(e)
                print(rel_file_location)

            bar.update(count)
    bar.finish()

    return seizure_data_dict

# to convert raw edf data into pkl format raw data
def gen_raw_seizure_pkl(args, tuh_eeg_szr_ver, anno_file):
    
    base_dir = args.base_dir
    merge_train_dev = args.merge_train_dev

    # Create the directory where the raw seizure data will be extracted
    save_data_dir = os.path.join(base_dir, tuh_eeg_szr_ver, 'raw_seizures_test')
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
        
    if merge_train_dev :
        save_data_dir = os.path.join(save_data_dir, 'merged')
        # Same folder 
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
    else :
        # Create one folder for dev and another for train
        save_data_dev = os.path.join(save_data_dir, 'dev')
        save_data_train = os.path.join(save_data_dir, 'train')
        if not os.path.exists(save_data_dev):
            os.makedirs(save_data_dev)
        if not os.path.exists(save_data_train):
            os.makedirs(save_data_train)

    raw_data_base_dir = os.path.join(base_dir, tuh_eeg_szr_ver)
    szr_annotation_file = os.path.join(raw_data_base_dir, '_DOCS', anno_file) # excel file with the seizure info for each recording

    print('\n',szr_annotation_file,'\n')

    # For training files
    print('Parsing the seizures of the training set...\n')
    train_data_dict = generate_data_dict(szr_annotation_file, 'train', tuh_eeg_szr_ver)
    print('Number of seizures by type in the training set...\n')
    print_type_information(train_data_dict)
    print('\n\n')

    # For dev files
    if tuh_eeg_szr_ver == 'v1.5.2':
        dev_name = 'dev'
    else:
        exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

    print('Parsing the seizures of the validation set...\n')
    dev_test_data_dict = generate_data_dict(szr_annotation_file, dev_name, tuh_eeg_szr_ver)
    print('Number of seizures by type in the validation set...\n')
    print_type_information(dev_test_data_dict)
    print('\n\n')

    if tuh_eeg_szr_ver == 'v1.5.2' :
        if merge_train_dev :
            # Now we combine both
            print('Combining the training and validation set...\n')
            merged_dict = merge_train_test(dev_test_data_dict, train_data_dict)
            # merged_dict = merge_train_test(train_data_dict,dev_test_data_dict)
            print('Number of seizures by type in the combined set...\n')
            print_type_information(merged_dict)
            print('\n\n')

            # Extract the seizures from the edf files and save them
            print('Extracting seizures from both dev and train directories...\n')
            seizure_data_dict = load_edf_extract_seizures_v152(raw_data_base_dir, save_data_dir, merged_dict)
            print('\n\n')
        else :
            # Separately extract the seizures from the def and train edf files and save them
            print('Extracting seizures from the dev directory...\n')
            seizure_data_dict = load_edf_extract_seizures_v152(raw_data_base_dir, save_data_dev, dev_test_data_dict)
            print('\n\n')
            print('Extracting seizures from the train directory...\n')
            seizure_data_dict = load_edf_extract_seizures_v152(raw_data_base_dir, save_data_train, train_data_dict)
            print('\n\n')
    else:
        exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

    print_type_information(seizure_data_dict)
    print('\n\n')

def main():

    parser = argparse.ArgumentParser(description='Build data for TUH EEG data')
    
    parser.add_argument('--base_dir', default='../data', help='path to seizure dataset')
    parser.add_argument('-v', '--tuh_eeg_szr_ver', default='v1.5.2', help='version of TUH seizure dataset')
    parser.add_argument('--merge_train_dev', default=False, help='boolean to keep dev and train separate')

    args = parser.parse_args()
    parser.print_help()

    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver

    if tuh_eeg_szr_ver == 'v1.5.2': # for v1.5.2
        anno_file = 'seizures_v36r.xlsx'
        gen_raw_seizure_pkl(args, tuh_eeg_szr_ver, anno_file)
    else:
        exit('Not supported version number %s'%tuh_eeg_szr_ver)
    

if __name__ == '__main__':
    main()
