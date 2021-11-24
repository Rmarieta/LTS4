#!/bin/sh


echo "BAM"
read -s -p "Password:" password
echo "$password"

expect "nedc@www.isip.piconepress.com's password:"
send "$password"

/usr/bin/expect <<EOD
spawn ssh -oStrictHostKeyChecking=no -oCheckHostIP=no usr@$myhost.example.com
expect "password"
send "$PWD\n"
interact
EOD
echo "you're out"