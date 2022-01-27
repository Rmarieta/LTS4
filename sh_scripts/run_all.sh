#!/bin/bash

# Run this script to run everything on an unlimited loop, Ctrl+C to stop it
# If not working, check the path of the tug_eeg_seizure dataset, it may have been modified

while :
do

    expect rsync_answer.exp eeg/tuh_eeg_seizure/v1.5.2 ../data nedc_resources

done