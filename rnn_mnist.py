from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

log_dir = 'data/tf_logs'
model_dir = 'data/tf_models'
timestamp = int(time.time())

# Dimensions:
# --- X_train(60000, 28, 28)    --- y_train(60000,)
# --- X_test(10000, 28, 28)     --- y_test(10000,)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalization
X_train = X_train / 255
X_test = X_test / 255

# TensorBoard logging for visualization of the results
tensorboard = TensorBoard(log_dir = f'{log_dir}/{timestamp}')

# Model architecture
model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer = Adam(lr=0.001, decay=1e-6), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# Training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard])
val_acc = history.history["val_acc"][-1]
val_loss = history.history["val_loss"][-1]

# Saving the model for later use
model.save(f'{model_dir}/{timestamp}.h5')
# model = load_model(f'{model_dir}/1557517060.h5')

# Finding out which predictions have been wrong
predictions = model.predict_classes(X_test, batch_size=32, verbose=1)
wrong_predictions = np.nonzero(predictions != y_test)[0]       # gives indices of unequal values
np.random.shuffle(wrong_predictions)                           # randomizing wrong predictions order

# Plot settings
h, w = 10, 10        # for raster image
nrows, ncols = 5, 4  # array of sub-plots
figsize = [6, 8]     # figure size, inches

# Create figure (fig), and array of axes (ax)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

# Plot simple raster image on each sub-plot
for i, axi in enumerate(ax.flat):
    # i runs from 0 to (nrows*ncols-1) = 19
    # axi is equivalent with ax[rowid][colid]
    index_wrong_pred = wrong_predictions[i]
    img = cv2.resize(X_test[index_wrong_pred], (10,10))
    axi.imshow(img)
    # get indices of row/column
    # write row/col indices as axes' title for identification
    axi.set_title(f'P: {predictions[index_wrong_pred]}, Y: {y_test[index_wrong_pred]}')
    axi.axis('off')

plt.tight_layout(True)
plt.show()

# Printing the evaluation score of the test set
print('\nTest result: %.3f loss: %.3f' % (val_acc*100,val_loss))
