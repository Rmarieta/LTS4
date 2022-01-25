# Shell Scripts to Run on the Cluster

The scripts provided here are examples of files that can be used to run the different python scripts on the cluster. To run one of them, run the command (at the level of <code>LTS4/</code>) :
```
sbatch scitas_run/example_file.sh
```
Feel free to change the time and the memory (RAM) variables on top of each files according to the needs of the script. Allocate at least (70'000 of memory if you want to compute the Laplacian).

Once you run one of those, it starts a job with an associated number (job_number). You can cancel a job anytime by running :
```
scancel job_number
```
The terminal output will be saved in the output file selected in the shell script under the line <code>#SBATCH --output ./cluster_output/build_data.out</code>. For convenient use and potential debugging, it is desirable to open a new terminal and run :
```
tail -f cluster_output/build_data.out
```
to keep the output file <code>build_data.out</code> in reading mode and not having to open it again after every change. 

The line <code>#SBATCH --chdir /folder</code> will have to be adapted depending on where the script is run and on your username on the cluster. If it is run on <code>scratch/</code> see how it is done in <code>scratch_build_data.sh</code>, otherwise, check the other files.

The line <code>source ../../../home/rmarieta/rma_env/bin/activate</code> will also need to be adapted depending on your username and where you're running the script from. This line activates the virtual environment, which is located in <code>home/rmarieta/rma_env/bin/activate</code>.