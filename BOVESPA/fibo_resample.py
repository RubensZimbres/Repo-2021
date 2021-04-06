# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import datetime
import pandas as pd
from pandas_datareader import data
import numpy as np
from datetime import timedelta
#pip install pandas-datareader
import time
start=time.time()
hoje =  (datetime.datetime.today() - timedelta(days=10))      ##datetime.datetime.today()

inicio=(datetime.datetime.today() - timedelta(days=320))
data1=str(hoje.year)+'-'+str(hoje.month)+'-'+str(hoje.day)
hoje

# %% [markdown]
# 

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:49:27 2020

@author: theone
"""
import datetime
import pandas as pd
from pandas_datareader import data
import numpy as np
from datetime import timedelta
#pip install pandas-datareader
import time
start=time.time()

inicio=(datetime.datetime.today() - timedelta(days=320))#.strftime("%Y,%m,%d")

def RSI(dataset, n=14):
    delta = dataset.diff()
    #-----------
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = pd.Series(dUp).rolling(window=n).mean()
    RolDown = pd.Series(dDown).rolling(window=n).mean().abs()
    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    return rsi



def predict(acao):

    try:
        if acao=='USDBRL=X' or acao=='^BVSP' or acao=='GC=F' or acao=='CL=F' or acao=='ZS=F' or acao=='SI=F' or acao=='^DJI' or acao=='^IXIC':
            stock=acao
        else:
            stock = '{}.SA'.format(acao)
        source = 'yahoo'
        
        # Set date range (Google went public August 19, 2004)
        start =inicio#datetime.datetime(2020, 3, 1)
        end = hoje
        
        # Collect Google stock data
        goog_df = data.DataReader(stock, source, start, end)
        
        dataset = goog_df['Adj Close'].resample('W').mean()
        print(dataset.head(5))

        #goog_df['Adj Close'].plot(kind='line', grid=True, title='{} Adjusted Closes, IPO through 2016'.format(stock))
        
        ## FIBO DINAMICO
        max_periodo=np.max(dataset[-144:])
        min_periodo=np.min(dataset[-144:])
        
        diff=max_periodo-min_periodo
        
        primeiro_fibo=min_periodo+0.328*diff
        segundo_fibo=min_periodo+0.618*diff
        meio=min_periodo+0.5*diff
        media_verde=np.mean(dataset[-3:])
        media_vermelha=np.mean(dataset[-13:])
        
        tendencia=media_verde-media_vermelha
        print(acao,tendencia)
        fibo_superior=dataset[-1]-segundo_fibo
        #limiar_superior=(max_periodo-segundo_fibo)
        #limiar_inferior=(primeiro_fibo-min_periodo)
        
        
        fibo_inferior=dataset[-1]-primeiro_fibo
        
        rsi=RSI(goog_df.Close)[-1]

        if tendencia>0 and fibo_inferior>0 and  dataset[-1]>dataset[-5] and dataset[-1]>primeiro_fibo:
            output='buy'
            print('buy')
        elif tendencia<0 and fibo_superior<0 and dataset[-1]<dataset[-5] and dataset[-1]<segundo_fibo:
            output='short'
            print('short')
        else:
            output='wait'
            print('wait')
        close=dataset[-1]
        ganho=(max_periodo/segundo_fibo)-1
        diferenca=media_verde-media_vermelha
        oportunidade=dataset[-1]-meio
        momentum=((dataset[-1]-dataset[-11])/dataset[-1])*1000
    except Exception as e:
        print(e)
        output='NA'
        close='NA'
        ganho='NA'
        diferenca='NA'
        oportunidade='NA'
        rsi='NA'
        momentum='NA'
    return output, close,ganho,diferenca,oportunidade, rsi, momentum

df=pd.read_csv('Lista_Acoes_Setor.csv',sep=',',header=0)
df.columns=['Sigla','Nome_Empresa','Setor']
df['Previsao']=np.zeros(df.shape[0])
df['Close']=np.zeros(df.shape[0])
df['Potencial']=np.zeros(df.shape[0])
df['MM_diff']=np.zeros(df.shape[0])
df['eixo_xx']=np.zeros(df.shape[0])
df['Data']=np.zeros(df.shape[0])
df['IFR']=np.zeros(df.shape[0])
df['Momentum']=np.zeros(df.shape[0])



# %%
for i in range(0,df.shape[0]):
    try:#df.shape[0]):
        prev=predict(df.iloc[i,0])
        df['Previsao'].iloc[i]=prev[0]
        df['Close'].iloc[i]=prev[1]
        df['Potencial'].iloc[i]=prev[2]
        df['MM_diff'].iloc[i]=prev[3]
        df['eixo_xx'].iloc[i]=prev[4]
        df['IFR'].iloc[i]=prev[5]
        df['Data'].iloc[i]=data1
        df['Momentum'].iloc[i]=prev[6]
    except:
        pass
df['Mood']=np.zeros(df.shape[0])

for i in range(0,df.shape[0]):
    if df['Previsao'][i]=='buy':
        df['Mood'][i]=1
    if df['Previsao'][i]=='short':
        df['Mood'][i]=-1
    if df['Previsao'][i]=='wait':
        df['Mood'][i]=0
    if df['Previsao'][i]=='NA':
        df['Mood'][i]=0

# %% [markdown]
# 

# %%
def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

for i in range(0,df.shape[0]):
    if df.Previsao[i]=='buy':
        df['Mood'][i]=1
    elif df.Previsao[i]=='short':
        df['Mood'][i]=-1
    else:
        df['Mood'][i]=0
        
        


# %%
df=df[df.Potencial!='NA']


# %%
df['Y_ForçaMM']=norm(df.Potencial)
df['X_Probabilidade_Fibo']=norm(df.eixo_xx)
df['Momentum']


# %%
df['Momentum']-float(df[df['Sigla']=='^BVSP']['Momentum'])


# %%
#df['Momentum']=df['Momentum']-float(df[df['Sigla']=='^BVSP']['Momentum'])
df['IFR']=df['IFR']-float(df[df['Sigla']=='^BVSP']['IFR'])
df['IFR']


# %%
df['Momentum']=df['Momentum']/280
df['Momentum']


# %%
df[['Sigla', 'Nome_Empresa', 'Previsao', 'Close', 'Potencial', 'MM_diff',
       'eixo_xx', 'Mood', 'Y_ForçaMM', 'X_Probabilidade_Fibo','Data','Setor','IFR','Momentum']].to_csv('dataframe_PowerBI.csv',sep=',',index=False,columns=df.columns)


# %%
import datetime

import pandas as pd
import pytz
import numpy as np
# Construct a BigQuery client object.

# TODO(developer): Set table_id to the ID of the table to create.
# table_id = "your-project.your_dataset.your_table_name"


dataframe = pd.read_csv('dataframe_PowerBI.csv',sep=',',header=0)

dataframe=dataframe.dropna(axis=0)
dataframe.columns
dataframe


# %%
dataframe['Data'] = dataframe['Data'].astype('datetime64[ns]')


# %%
dataframe.columns


# %%
dataframe.columns=['Sigla', 'Nome_Empresa', 'Setor', 'Previsao', 'Close', 'Potencial',
       'MM_diff', 'eixo_xx', 'Data', 'IFR', 'Momentum', 'Mood','Y_For__aMM',
       'X_Probabilidade_Fibo']
dataframe=dataframe[['Sigla', 'Nome_Empresa', 'Previsao', 'Close', 'Potencial', 'MM_diff',
       'eixo_xx',  'Momentum', 'Y_For__aMM', 'X_Probabilidade_Fibo','Data','Setor','IFR','Mood']]
dataframe


# %%
from google.cloud import bigquery
client = bigquery.Client()

job_config = bigquery.LoadJobConfig(
    # Specify a (partial) schema. All columns are always written to the
    # table. The schema is used to assist in data type definitions.
    schema=[
        # Specify the type of columns whose type cannot be auto-detected. For
        # example the "title" column uses pandas dtype "object", so its
        # data type is ambiguous.
        bigquery.SchemaField("Sigla", bigquery.enums.SqlTypeNames.STRING),
        # Indexes are written if included in the schema by name.
        bigquery.SchemaField("Nome_Empresa", bigquery.enums.SqlTypeNames.STRING),
        bigquery.SchemaField("Previsao", bigquery.enums.SqlTypeNames.STRING),
        bigquery.SchemaField("Close", bigquery.enums.SqlTypeNames.FLOAT),
        bigquery.SchemaField("Potencial", bigquery.enums.SqlTypeNames.FLOAT),
        bigquery.SchemaField("MM_diff", bigquery.enums.SqlTypeNames.FLOAT),
        bigquery.SchemaField("eixo_xx", bigquery.enums.SqlTypeNames.FLOAT),
        bigquery.SchemaField("Mood", bigquery.enums.SqlTypeNames.INTEGER),
        bigquery.SchemaField("Y_For__aMM", bigquery.enums.SqlTypeNames.FLOAT),
        bigquery.SchemaField("X_Probabilidade_Fibo", bigquery.enums.SqlTypeNames.FLOAT),
        bigquery.SchemaField("Data", bigquery.enums.SqlTypeNames.DATE),
        bigquery.SchemaField("Momentum", bigquery.enums.SqlTypeNames.FLOAT),
        bigquery.SchemaField("Setor", bigquery.enums.SqlTypeNames.STRING),
        bigquery.SchemaField("IFR", bigquery.enums.SqlTypeNames.FLOAT),
    ],
    # Optionally, set the write disposition. BigQuery appends loaded rows
    # to an existing table by default, but with WRITE_TRUNCATE write
    # disposition it replaces the table with the loaded data.
    write_disposition="WRITE_APPEND",
)

job = client.load_table_from_dataframe(
    dataframe, 'machinelearning-XXX.fiboXXX.XXXnha', job_config=job_config
)  # Make an API request.
job.result()  # Wait for the job to complete.


# %%
np.min(df.Momentum)


# %%



