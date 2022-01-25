# Download the Dataset

In case you're either on Linux or on the SCITAS EPFL servers, you can use the provided scripts to automate the download of the dataset (v1.5.2). 
Downloading the dataset requires a password (<code>nedc_resources</code>) that you can acquire through the website. The download sometimes crashes and to avoid having to re-enter the password multiple times, a <code>run_all.sh</code> script is provided. 

<code>run_all.sh</code> can be executed running :
```
sh run_all.sh
```
You might first need to execute the permission to run these shell scripts with :
```
chmod u+r+x run_all.sh
```

 Same for <code>rsync_nedc.sh</code>. <code>run_all.sh</code> runs an infinite loop that has to be manually stopped with <code>Ctrl+C</code> once the download is finished. This is useful in case the password problem occurs. Maybe first try to remove the while loop and only leave <code>expect rsync_answer.exp tuh_eeg_seizure/v1.5.2 data/v1.5.2 nedc_resources</code> in the script. If it works, no need to run the loop. The dataset will be saved in data/v1.5.2.

The path to the dataset might change, as it already happened once in the last 6 months, so you might have to make sure the <code>tuh_eeg_seizure/v1.5.2</code> argument in the except command is up to date.

If you're working on the cluster, make sure you're aware of the storage limits on the different folders : <code>scratch/</code> has no limits but your data can be removed without your consent, while <code>home/</code> is backed up daily but has a limit of 100GB. The dataset takes about 60GB already, and the extracted seizures (selecting FNSZ, GNSZ and BG) take up about 70GB which is already over the limit. So it is preferred to download the dataset to <code>scratch/</code> and then extract the seizures to <code>home/</code> so you can do the rest of the work on <code>home/</code>.