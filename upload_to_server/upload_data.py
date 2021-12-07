"""
Currently, only Fidis has a serial partition with a pay-as-you-use policy. On the other clusters, you will be charged for the whole nodes even if you use only a fraction of them.

Plan : 
- upload data/v1.5.2/edf & data/v1.5.2/_DOCS to scratch/rmarieta (potential removal every 2 weeks)
- output all results to home/rmarieta (100GB limit), for example in home/rmarieta/raw_samples

To connect to the cluster, run :
ssh -X rmarieta@fidis.epfl.ch


Run the following command (from local device) to copy the script from local to remote (upload_data.py and run.sh will be in /home/rmarieta/)
scp upload_data.py run.sh rmarieta@fidis.epfl.ch:~/
To transfer a .tar file from the cluster to the local device (on local)
scp rmarieta@fidis.epfl.ch:~/LTS4/data/v1.5.2/data.tar .
To unpack it :
tar -xvf data.tar

After that, run (in rmarieta/) :
sbatch run.sh

Problem of compatibility between Windows and Linux
=> run : dos2unix.sh in the terminal in upload_to_server/ to adapt the run.sh file before copying it to the 

In run.sh : 2 options
1) Saving the output in scratch/ :
    #SBATCH --chdir /scratch/rmarieta
    and
    python ../../home/rmarieta/upload_data.py
2) Saving the output in home/ :
    #SBATCH --chdir /home/rmarieta
    and
    python upload_data.py

To build a .tar file : tar -zcvf data.tar ./data/v1.5.2/edf ./data/v1.5.2/_DOCS
To upload it on the cluster : scp data.tar rmarieta@fidis.epfl.ch:/scratch/rmarieta
To unpack it on the cluster (in /scratch/rmarieta) : tar -xvf filename.tar

To open text editor in terminal : vi file.py, vim file.py, nano file.py, or code file.py
To delete directory : rm -r path/to/dir (e.g rm -r data in scratch/rmarieta will delete data)
To delete all files inside a directory : rm path/to/dir
To delete all files in current directory + ask for confirmation : rm -i *

To create the repository data/v1.5.2 : 
mkdir data
mkdir data/v1.5.2

To download the dataset straight from the website :
rsync -auxvL nedc@www.isip.piconepress.com:data/tuh_eeg_seizure/v1.5.2/ .
password : nedc_resources

Use nedc_rsync.sh to automatise the process
Make sure to allow permission (on the cluster) running chmod u+r+x nedc_rsync.sh
Then run (in v1.5.2) : ../../nedc_rsync.sh nedc@www.isip.piconepress.com:data/tuh_eeg_seizure/v1.5.2/ .
"""

import os

dir = "data"

if not os.path.exists(dir) :
    os.makedirs(dir)