#!/bin/python

import sys
import socket
from datetime import datetime

# Target
print(sys.argv)
print("\n")
if len(sys.argv)==2:
	target="177.44.35.96" #socket.gethostbyname(sys.argv[1]) # translate hostname to ipv4
else:
	print("Invalid amount of args")

# Banner
print("-"*50)
print("Scanning target "+target)
print("Time started: "+ str(datetime.now()))
print("-"*50)

try:
	for port in range(1,65535):
		s=socket.socket(socket.AF_INET,socket.SOCK_STREAM) ## ipv4 + port
		socket.setdefaulttimeout(0.78)
		#print("Scanning port ", port)
		result=s.connect_ex((target,port))
		if result==0:
			print("Port {} is open".format(port))
		s.close()
except KeyboardInterrupt:
	print("Aborted")
	sys.exit()
	
except ocket.gaierror:
	print("Hostname could not be resolved")
	sys.exit()
	
except socket.error:
	print("Could not connect to server")
	sys.exit()
	
#python3 scanner.py <ip>
