#!/bin/bash

# Run this script to run everything on an unlimited loop, Ctrl+C to stop it
# Might be better to try :
# expect rsync_answer.exp tuh_eeg_seizure/v1.5.2 LTS4/data nedc_resources

while :
do

    expect rsync_answer.exp tuh_eeg_seizure/v1.5.2 data/v1.5.2 nedc_resources

done