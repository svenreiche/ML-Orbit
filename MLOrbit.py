
import tensorflow as tf

import matplotlib.pyplot as plt
import Model

class MLOrbit:
    def __init__(self):
        self.data = Model.Model()
        print('Tensorflow Version',tf.__version__)

    def prepareData(self):
        self.data.updateModelFromMachine()
        self.data.trackModel({})
        self.data.prepareData(100000, [0.02, 0.002, 0.02, 0.002, 0.1])   # jitter in x, xp,y,yp,delta in mm or 0.1%

    def runTF(self):
        n = 75000
        x_train = self.data.xdata[0:n,:]
        y_train = self.data.ydata[0:n,:]
        x_test = self.data.xdata[n:,:]
        y_test = self.data.ydata[n:,:]
        
        model = tf.keras.models.Sequential([
#            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(72, activation='relu',input_shape=x_train.shape[1:]),
 #           tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(5)
        ])

#        loss_fn = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer='adam',
                      loss="mean_squared_error",
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10)
        model.evaluate(x_test,  y_test, verbose=2)
        model.summary()
        y_predict=model.predict(x_test)
        plt.scatter(y_test[:,4],y_predict[:,4],s=0.5)
        plt.xlabel('Input Energy Jitter')
        plt.ylabel('Reconstructed Energy Jitter')
        plt.show()
        plt.scatter(y_test[:,4],x_test[:,5],s=0.5)
        plt.xlabel('Input Energy Jitter')
        plt.ylabel('Reading of first BPM in SARCL02')
        plt.show()



    def testTF(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        print(x_train.shape,x_test.shape)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        his=model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test,  y_test, verbose=2)
 
