#!/bin/bash

url=$1

if [ ! -d "$url" ] ;then 
	mkdir $url
fi
if [ ! -d "$url/recon" ] ;then 
	mkdir $url/recon
fi

echo "[+] Harvesting subdomains ..."

assetfinder $url >> $url/recon/assets.txt
cat $url/recon/assets.txt | grep $1 >> $url/recon/final.txt
rm $url/recon/assets.txt

echo "[+] Harvesting subdomains with amass ..."

amass enum -d $url >> $url/recon/f.txt

sort -u $url/recon/f.txt >> $url/recon/final.txt
rm $url/recon/f.txt

echo "[+] Probing for alive domains"
