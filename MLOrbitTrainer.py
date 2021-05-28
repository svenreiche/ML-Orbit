
import tensorflow as tf



class Trainer:
    def __init__(self):
        print('Tensorflow Version',tf.__version__)
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    def predict(self, x_data):
        return self.model.predict(x_data)

    def runTF(self, x_train, y_train, x_test, y_test, nepochs):

#        strategy = tf.distribute.MirroredStrategy()
#        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        nin = len(x_train[0,:])
        nout = len(y_train[0,:])
        print('Input size of first DNN layer:', nin)
        print('Output size of last DNN layer:', nout)
        print('Data size:',x_train.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

 #       with strategy.scope():
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(nin, activation='relu',input_shape=(nin,)),
            tf.keras.layers.Dense(nout)
        ])
        self.model.compile(optimizer='adam',
                          loss="mean_squared_error",
                          metrics=['accuracy'])

        self.history = self.model.fit(train_dataset, epochs=nepochs, validation_data=val_dataset)
        self.predict = self.model.predict(val_dataset)
        return
 #       return y_predict, history.history['loss'], history.history['accuracy'],\
 #              history.history['val_loss'], history.history['val_accuracy']

    def runTFInv(self, x_train, y_train, x_test, y_test, nepochs):
        nin = len(x_train[0,:])
        print('Input size of first DNN layer:', nin)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(nin, activation='relu',input_shape=x_train.shape[1:]),
#            tf.keras.layers.Dense(20, activation='relu'),
#            tf.keras.layers.Dense(10, activation='relu'),
#            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(72)
        ])

        self.model.compile(optimizer='adam',
                      loss="mean_squared_error",
                      metrics=['accuracy'])
        history = self.model.fit(x_train, y_train, epochs=nepochs,validation_data=(x_test, y_test))
        y_predict = self.model.predict(x_test)
        return y_predict, history.history['loss'], history.history['accuracy'],\
               history.history['val_loss'], history.history['val_accuracy']

#



