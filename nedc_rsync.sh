#!/bin/sh
#
# file: nedc_rsynch.sh
#
# this script detects an error with rsync and automatically restarts
# rsync if an error occurs. It is used to keep rsync running over
# long periods of time when you might regularly lose your network conneciton.
#
# Usage: nedc_rsync.sh user@host:/path...
#
# Example:
#  nedc_rsync.sh nedc_tuh_eeg@www.isip.piconepress.com:~/data/tuh_eeg/ .
#

# set up an infinite loop
#
RC=1
#read -s -p "Password:" password
while [[ $RC -ne 0 ]]
do

    # execute your rsync command
    echo "starting rsync..."
    #sshpass -p "nedc_resources" rsync -auxvL $1 .
    rsync -auxvL $1 .
    #expect "nedc@www.isip.piconepress.com's password:"
    #send "$password\n"
    RC=$?

    # display an informational message and sleep for a bit
    #
    echo "done with rsync..."
    sleep 1

done

#
# exit gracefully
