#!bin/python

import socket

HOST='127.0.0.1'
PORT = 8080

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  #### ipv4 and port

s.connect((HOST,PORT))

## nc -nvlp 8080 listening
