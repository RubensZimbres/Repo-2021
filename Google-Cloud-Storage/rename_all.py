import os
import re

a=[]
f = open("/home/rubens/.../FASE3/todos.txt", "r")
for x in f:
  a.append(x.replace('\n',''))

for i in range(0,len(a)):
    try:
        d=a[i].replace('LIGAÇÃO','LIGA').replace(' ','')
        c=a[i]
        os.system("gsutil mv '{0}' '{1}'".format(c,d))
    except:
        pass
