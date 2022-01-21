# Graph Computation

<code>covariance.py</code> and <code>laplacian.py</code> are functioning the same way, they both read through the seizure samples of the selected types, extract the signals, learn a graph from it and finally save all the graphs into .npy files (one graph per file). You can then use <code>plot_graphs.py</code> to visualise the graphs very easily. Note that this is not possible on the cluster unless you save the figures after plotting and then re-open them. So you might need to move your graphs on your local device after the graph computation step. To do so, run the 3 following commands where you saved the computed graphs :

- Tar the graph folder (on the cluster) : <code>tar -zcvf graph.tar ./graph_folder</code> 
- Copy <code>graph.tar</code> on your local (run on your local terminal) : <code>scp rmarieta@fidis.epfl.ch:~/LTS4/data/v1.5.2/graph.tar ./choose_output_path</code>
- Unpack the copied tar file on local : <code>tar -xvf graph.tar</code>

When running the two scripts, the following options are available :
- <code>--input_dir</code> : path to raw seizure samples.
- <code>--seizure_types</code>.
- <code>--graph_dir</code> : path to the output of the graph computation.
- <code>--chop</code> : to compute one graph per 1 second of a seizure sample instead of one graph per seizure sample.
- <code>--low_pass</code> : to low-pass filter the signal prior to computing the graph, you can also change the cut-off frequency of the filter in the code.

<code>scitas_run/cov.sh</code> and <code>scitas_run/laplacian.sh</code> provide an example on how to run the two scripts on the cluster. If not on the cluster, you can for example run the following command : <code>python learn_graph_laplacian/laplacian.py --graph_dir './data/v1.5.2/graph_lapl_lowp_50hz' --low_pass 'True' --seizure_types 'FNSZ' 'GNSZ' --chop 'False' </code> in <code>LTS4/</code>.