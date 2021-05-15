ping 192.168.0.1 -c 1 > ip.txt
cat ip.txt | grep "64 bytes"
192.168.0.ALL
cat ip.txt | grep "64 bytes" | cut -d " " -f 4   (delimiter and 4 spaces)
cat ip.txt | grep "64 bytes" | cut -d " " -f 4 | tr -d ":"  (translate delimiter)
mousepad ipsweep.sh

#!/bin/bash
for x in `seq 1 254`; do
ping -c 1 192.168.0.$x | grep "64 bytes" | cut -d " " -f 4 | tr -d ":" &
done

ipsweep.sh $1 $2
ex: ping -c 1 $1.$x | grep "64 bytes" | cut -d " " -f 4 | tr -d ":" &
chmod +x ipsweep.sh

#!/bin/bash
if [ "$1" ==""]
then
echo " input an IP address 192.168.0 bla bla bla"
else
for x in `seq 1 254`; do
ping -c 1 $1.$x | grep "64 bytes" | cut -d " " -f 4 | tr -d ":" &   ((; run one at a time))
done
fi

./ipsweep.sh >> ips.txt
for ip in $(cat ips.txt); do nmap $ip; done
