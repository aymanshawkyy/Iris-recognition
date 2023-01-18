from tensorflow import keras
import pickle


pickle_in = open("X3.pickle", "rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y3.pickle", "rb")
y_test = pickle.load(pickle_in)

X_test = X_test/255.0


model = keras.models.load_model('wh.model')

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])