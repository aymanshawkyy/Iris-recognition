import pickle
import neptune.new as neptune
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split
tf.random.set_seed(1234)

pickle_in = open("X1.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y1.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

model = keras.Sequential()

model.add(Flatten(input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
# fit model
history = model.fit(X, y, batch_size = 10, epochs=10, verbose=1)
# evaluate the model
model.save("wh.model")