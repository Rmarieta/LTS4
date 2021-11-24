
import dill as pickle
import matplotlib.pyplot as plt
import collections


import matplotlib.pyplot as plt

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])


# Run : python data_preparation/plot_data.py (in VS_code_workspace\EEG_seizure\seizure-type-classification-tuh-master)

print('\n')

N1 = r'..\data\v1.5.2\output\fft\fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12/szr_0_pid_00000258_type_TCSZ.pkl'
N2 = r'..\data\v1.5.2\output\fft\fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12/szr_1000_pid_00011333_type_FNSZ.pkl'
N3 = r'..\data\v1.5.2\output\fft\fft_seizures_wl2_ws_1.0_sf_250_fft_min_1_fft_max_12/szr_0_pid_00000258_type_TCSZ.pkl'

N4 = r'..\data\v1.5.2\raw_seizures\szr_3044_pid_00002448_type_ABSZ.pkl'


names = [N4]

for name in names :

    with open(name, "rb") as f:
        x = pickle.load(f)
        tab = x.data

    print(tab[0,:].shape)

    x1 = range(len(tab[0,:]))

    L = range(len(tab[:,0]))

    fig, axs = plt.subplots(len(L))

    for i in L :
        axs[i].plot(x1,tab[i,:])
        axs[i].axis('off')

plt.show()

print('\n')