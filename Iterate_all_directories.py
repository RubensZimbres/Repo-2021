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
