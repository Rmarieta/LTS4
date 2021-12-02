#!/bin/bash
​
RC=1
while [ $RC != 0 ]; do
	rsync -auxvL nedc@www.isip.piconepress.com:data/$1 $2
	RC=$?
	if [ $RC != 0 ]; then
		echo "Failed retrying in 10sec"
		sleep 10
	else
		echo "Done!"
	fi
done
