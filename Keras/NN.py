from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.metrics as km
from tensorflow.keras.layers import Dense, Dropout, Activation

batch_size=100
epochs = 100
learning_rate = 0.01

model = Sequential()
model.add(LSTM(12,input_shape=(1,look_back), kernel_initializer='uniform'))
model.add(Activation('relu'))
#model.add(Dense(32, kernel_initializer='uniform'))
#model.add(Activation('relu'))
model.add(Dense(1))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False,)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])
model.fit(trainX.reshape(290,1,4), trainY, epochs=epochs, batch_size=100, verbose=2)
score, acc = model.evaluate(testX.reshape(-1,1,4), testY,
                            batch_size=100)

#########################################


inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.Dense(64)(inputs)
lstm_out2 = keras.layers.Dense(64)(inputs)
outputs = keras.layers.Dense(1)(lstm_out2)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()
path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)
history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")
