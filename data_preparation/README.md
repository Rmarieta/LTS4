
# Seizure Extraction

<code>parameters.csv</code> contains the sampling frequency and the channel mapping used to extract the recordings and turn them into arrays of 20 signals.

<code>build_data.py</code> extracts all the .edf files from all folders and extracts them according to their seizure type into pickle format.
To do so, the script reads through the entries of the excel file located in <code>data/v1.5.2/_DOCS/seizures_v36r.xlsx</code>. This file contains all the seizure times as well as their types, which allows cropping of the recordings into seizure samples. You can select the output directory in which you want to save the seizure crops with <code>--output_dir</code>, and the seizure types that you wish to extract with <code>--seizure_types</code> (in the form <code>--seizure_types 'FNSZ' 'GNSZ' 'BG'</code> for example). Bear in mind that if you extract the background, it's going to take up a lot of space. The <code>'BG'</code> class is obtained by extracting the parts of the recordings with no seizure events (amongst the recordings that show at least one seizure event). The output will be saved in the output folder inside <code>raw_samples/</code>, and will be separated according to each seizure type. Each pickle file is saved in the form <code>raw_samples/file_0_pid_00005698_type_FNSZ.pkl</code> with the patient id of the patient that experienced this seizure.

To run it on the cluster, use <code>scitas_run/scratch_build_data.sh</code>, which will run the python script. Otherwise, run (<code>LTS4/</code>) :

```
python data_preparation/build_data.py --seizure_types 'FNSZ' 'GNSZ' 
```
