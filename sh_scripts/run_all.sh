#!/bin/bash

# Run this script to run everything on an unlimited loop, Ctrl+C to stop it
# Might be better to try (without the loop, but if it crashes run the following):
# expect rsync_answer.exp tuh_eeg_seizure/v1.5.2 LTS4/data nedc_resources
# If not working, check the path of the tug_eeg_seizure dataset, it may have been modified

while :
do

    expect rsync_answer.exp tuh_eeg_seizure/v1.5.2 data/v1.5.2 nedc_resources

done