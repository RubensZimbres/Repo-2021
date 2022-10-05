import pandas as pd
import os
import json

rootdir='/home/theone/Documents/Enterprise/data/'

os.listdir(rootdir)

locais=[]
for path, subdirs, files in os.walk(rootdir):
    for name in files:
        locais.append(os.path.join(path, name))

jsons = []
for arquivo in locais:
    for line in open(arquivo):
        jsons.append(json.loads(line))
pd.DataFrame.from_records(jsons).head()


--------------------------------------------


import pandas as pd
import os
import numpy as np
rootdir='/home/user/Downloads/other_models/prediction'

df=pd.read_csv(rootdir+'/'+sorted(os.listdir(rootdir))[1],sep=',',header=0)

df0=[]

for files in sorted(os.listdir(rootdir)):
    df0.append(pd.read_csv(rootdir+'/'+files,sep=',',header=0))
    print(files)

df0[0]

result_prev=pd.concat(df0,axis=0)
