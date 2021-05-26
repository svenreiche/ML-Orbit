import os
import sys
os.environ["EPICS_CA_ADDR_LIST"]="sf-cagw"
os.environ["EPICS_CA_SERVER_PORT"]="5062"

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)



from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.uic import loadUiType


import MLOrbitModel
import MLOrbitData
import MLOrbitTrainer

Ui_MLOrbitGUI, QMainWindow = loadUiType('MLOrbit.ui')


class MLOrbitGUI(QMainWindow, Ui_MLOrbitGUI):
    def __init__(self):
        super(MLOrbitGUI, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Machine Learning - Orbit Analysis")
        self.initmpl()
# included classes
        self.data = MLOrbitData.OrbitData()   # has machine BPM readings and magnet settings
        self.model = MLOrbitModel.OrbitModel() # interface to Madx to generate R-MAtrix and sampel orbits
        self.train = MLOrbitTrainer.Trainer()  # all Tensorflow stuff

        self.hasLoaded = False
        self.hasModel = False
        self.hasData = False
# event
        self.UILoadData.clicked.connect(self.LoadData)
        self.UIGenerateModel.clicked.connect(self.GenerateModel)
        self.UIGenerateTraining.clicked.connect(self.GenerateTraining)
        self.UITrainModel.clicked.connect(self.TrainModel)


    def TrainModel(self):
        if not self.hasData:
            self.GenerateTraining()

        nepoch = int(str(self.UITrainEpochs.text()))
        n = int(self.model.xdata.shape[0]*0.75)  # 75 percent are training and rest testing
        x_train = self.model.xdata[0:n, :]
        y_train = self.model.ydata[0:n, :]
        x_test = self.model.xdata[n:, :]
        y_test = self.model.ydata[n:, :]

        y_predict, loss, acc, val_loss, val_acc=self.train.runTF(x_train, y_train, x_test, y_test, nepoch)
        ep =np.linspace(1,nepoch,num=nepoch)

        self.axes1.clear()
        self.axes1.plot(ep,loss,label=r'Training Set')
        self.axes1.plot(ep,val_loss,label=r'Test Set')
        self.axes1.set_xlabel('Epoch')
        self.axes1.set_ylabel('Loss Function')
        self.axes1.legend()

        self.axes2.clear()
        self.axes2.plot(ep, acc, label=r'Training Set')
        self.axes2.plot(ep, val_acc, label=r'Test Set')
        self.axes2.set_xlabel('Epoch')
        self.axes2.set_ylabel('Accuracy')
        self.axes2.legend()

        self.axes3.clear()
        self.axes3.scatter(y_test[:,4],y_predict[:,4],s=0.5)
        self.axes3.set_xlabel(r'Input Energy Jitter (0.1%)')
        self.axes3.set_ylabel(r'Reconstructed Jitter (0.1%)')

        self.axes4.clear()
        self.axes4.scatter(y_test[:, 0], y_predict[:, 0], s=0.5)
        self.axes4.set_xlabel(r'Input Orbit Jitter in X (mm)')
        self.axes4.set_ylabel(r'Reconstructed Jitter (mm)')

        self.canvas.draw()



    def GenerateTraining(self):
        if not self.hasModel:
            self.GenerateModel()

        nsam = int(str(self.UITrainSamples.text()))
        errorb = float(str(self.UIOrbit.text()))*1e-3
        errang = float(str(self.UIAngle.text()))*1e-3
        errerg = float(str(self.UIEnergy.text()))
        errbpm = float(str(self.UIBPM.text()))*1e-3
        fluc=self.model.generateTrainingsData(nsam,[errorb,errang,errorb,errang,errerg,errbpm])
        self.hasData = True
        nx=int(fluc.shape[0]/2)

        self.axes1.clear()
        yold=fluc[0:nx,0]*0
        for i in range(5):
            ynew = np.sqrt(yold**2+fluc[0:nx,i]**2)
            self.axes1.fill_between(self.model.s,ynew,yold)
            yold = ynew

        self.axes2.clear()
        for i in range(5):
            self.axes2.plot(self.model.s, fluc[0:nx,i])

        self.axes3.clear()
        yold = fluc[nx:, 0] * 0
        for i in range(5):
            ynew = np.sqrt(yold ** 2 + fluc[nx:, i] ** 2)
            self.axes3.fill_between(self.model.s, ynew, yold)
            yold = ynew

        self.axes4.clear()
        for i in range(5):
            self.axes4.plot(self.model.s, fluc[nx:, i])

        self.canvas.draw()

    def GenerateModel(self):
        if not self.hasLoaded:
            self.LoadData()
        self.model.updateModel(self.data.mag, self.data.energy)
        self.hasModel = True
        self.axes1.clear()
        self.axes1.plot(self.model.s, self.model.rx1,label=r'$R_{11}$')
        self.axes1.plot(self.model.s, self.model.rx2, label=r'$R_{12}$')
        self.axes1.legend()

        self.axes2.clear()
        self.axes2.plot(self.model.s, self.model.ry3, label=r'$R_{33}$')
        self.axes2.plot(self.model.s, self.model.ry4, label=r'$R_{34}$')
        self.axes2.legend()

        self.axes3.clear()
        self.axes3.plot(self.model.s, self.model.rx5, label=r'$R_{16}$')
        self.axes3.plot(self.model.s, self.model.ry5, label=r'$R_{36}$')
        self.axes3.legend()

        self.axes4.clear()
        self.canvas.draw()

    def LoadData(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "DataMiner File",
                                                  "/sf/data/applications/BD-DataMiner/AramisOrbit/data",
                                                  "HDF5 Files (*.h5)", options=options)
        if fileName is None:
            return
        self.data.open(fileName)
        self.hasLoaded = True

# initializing matplotlib
    def initmpl(self):
        self.fig=Figure()
        self.axes1=self.fig.add_subplot(221)
        self.axes2=self.fig.add_subplot(222)
        self.axes3=self.fig.add_subplot(223)
        self.axes4=self.fig.add_subplot(224)
        self.canvas = FigureCanvas(self.fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar=NavigationToolbar(self.canvas,self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)
# Main  routine
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MLOrbitGUI()
    main.show()
    sys.exit(app.exec_())

