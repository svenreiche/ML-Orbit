
import tensorflow as tf



class Trainer:
    def __init__(self):
        print('Tensorflow Version',tf.__version__)


    def runTF(self, x_train, y_train, x_test, y_test, nepochs):

        nin = len(x_train[0,:])
        print('Input size of first DNN layer:', nin)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(nin, activation='relu',input_shape=x_train.shape[1:]),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(5)
        ])

        model.compile(optimizer='adam',
                      loss="mean_squared_error",
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=nepochs,validation_data=(x_test, y_test))
        y_predict = model.predict(x_test)
        return y_predict, history.history['loss'], history.history['accuracy'],\
               history.history['val_loss'], history.history['val_accuracy']

#





