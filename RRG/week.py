#pip install pandas-datareader
import datetime
import pandas as pd
from pandas_datareader import data
import numpy as np
from datetime import timedelta
import time
import os

pasta_save='/home/XXX/Shared/RRG/data_py/'

hoje =  (datetime.datetime.today() - timedelta(days=0))  
inicio=(datetime.datetime.today() - timedelta(days=320))

lista=['^BVSP','USIM5','OIBR4','TEND3','BBDC4','PETZ3','LAME4','MGLU3','WEGE3', 'PRIO3', 'VVAR3','JBSS3', 'MRFG3','RENT3','JHSF3', 'GGBR4','CSNA3',
'ELET6','MOVI3','PETR4','RAIL3']

def save_acao(parte):
    acao=lista[parte]
    if acao=='USDBRL=X' or acao=='^BVSP' or acao=='GC=F' or acao=='CL=F' or acao=='ZS=F' or acao=='SI=F' or acao=='^DJI' or acao=='^IXIC':
        stock=acao
    else:
        stock = '{}.SA'.format(acao)
    source = 'yahoo'
    goog_df = data.DataReader(stock, source, inicio, hoje).resample('W-MON').mean().reset_index()
    goog_df.columns=['Date','High', 'Low', 'Open', 'Close', 'Volume', 'Adj_Close']
    goog_df.Date=pd.to_datetime(goog_df.Date).dt.strftime('%m/%d/%Y')
    goog_df.to_csv(pasta_save+acao+'.csv',sep=';',index=False)
    return print(acao, 'OK')

list(map(save_acao,np.arange(0,len(lista))))
