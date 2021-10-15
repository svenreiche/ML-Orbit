
import tensorflow as tf



class Trainer:
    def __init__(self):
        print('Tensorflow Version',tf.__version__)
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


    def runTF(self, x_train, y_train, x_test, y_test, nepochs):

#        strategy = tf.distribute.MirroredStrategy()
#        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        nin = len(x_train[0,:])
        nout = len(y_train[0,:])
        print('Input size of first DNN layer:', nin)
        print('Output size of last DNN layer:', nout)
        print('Data size:',x_train.shape)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

 #       with strategy.scope():
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(nin, activation='relu',input_shape=(nin,)),
            tf.keras.layers.Dense(nout)
        ])
        self.model.compile(optimizer='adam',
                          loss="mean_squared_error",
                          metrics=['accuracy'])

        self.history = self.model.fit(self.train_dataset, epochs=nepochs, validation_data=self.val_dataset)
        self.predict = self.model.predict(self.val_dataset)
        return


    def runTFInv(self, x_train, y_train, x_test, nepochs):

        nin = len(x_train[0, :])
        nout = len(y_train[0, :])
        print('Input size of first DNN layer:', nin)
        print('Output size of last DNN layer:', nout)
        print('Data size:', x_train.shape)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

        #       with strategy.scope():
        self.modelInv = tf.keras.models.Sequential([
            tf.keras.layers.Dense(nin, activation='relu', input_shape=(nin,)),
            tf.keras.layers.Dense(nout)
        ])
        self.modelInv.compile(optimizer='adam',
                           loss="mean_squared_error",
                           metrics=['accuracy'])

        self.historyInv = self.modelInv.fit(self.train_dataset, epochs=nepochs)
        return self.modelInv.predict(x_test)





