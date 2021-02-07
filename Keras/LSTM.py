import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
#pip install pandas-datareader

stock = 'RENT3.SA'
source = 'yahoo'

# Set date range (Google went public August 19, 2004)
start = datetime.datetime(2005, 8, 19)
end = datetime.datetime(2019, 8, 6)

# Collect Google stock data
goog_df = data.DataReader(stock, source, start, end)

dataset = goog_df['Adj Close']

import tensorflow as tf
import pandas as pd  
import numpy as np
from  sklearn.preprocessing import MinMaxScaler

dataset = np.array(dataset.astype('float32')).reshape(-1,1)

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

#dataset=norm(dataset)

look_back=4
np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.99)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

def create_dataset(dataset, look_back=look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX

trainY = trainY.reshape(len(trainY), 1)
testY = testY.reshape(len(testY), 1)
trainY

X0=trainX
Y0=trainY

X0=X0.reshape(X0.shape[0],X0.shape[1],1)
testX=testX.reshape(testX.shape[0],testX.shape[1],1)


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class MyModel(tf.keras.Model):
  def __init__(self, rnn_units):
    super().__init__(self)
    self.gru = tf.keras.layers.LSTM(rnn_units,activation='tanh', recurrent_activation='sigmoid',
    use_bias=True, kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros', unit_forget_bias=True,
    kernel_regularizer=None,
                                   return_sequences=True, 
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(1)

  def call(self, inputs, states=None, return_state=False, training=False):
    x=inputs
    x, states, z = self.gru(x, initial_state=states, training=training)
    x = tf.nn.relu(x)
    pred = self.dense(x, training=training)
    
#    if return_state:
#      return pred, states
#    else: 
    return pred


rnn_units = 40

model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    rnn_units=rnn_units)


from tensorflow import keras
loss=tf.keras.losses.MeanSquaredError()

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=loss,metrics='mae')

checkpoint_dir = './training_checkpoints'
import os 
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=100

history = model.fit(X0,Y0, epochs=EPOCHS, callbacks=[checkpoint_callback])

import matplotlib.pyplot as plt 
plt.plot(np.mean(model.predict(testX),axis=1).reshape(1,-1)[0])
plt.plot(testY.reshape(1,-1)[0])
plt.show()

np.mean(abs(np.mean(model.predict(testX),axis=1).reshape(1,-1)[0]/testY.reshape(1,-1)[0]))
