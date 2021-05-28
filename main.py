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

# flags
        self.hasModel = False
        self.hasTraining = False
# event
        self.UIOpenFile.clicked.connect(self.OpenFile)
        self.UITrainModel.clicked.connect(self.TrainModel)
# plotting
        self.actionR11.triggered.connect(self.PlotR)
        self.actionR12.triggered.connect(self.PlotR)
        self.actionR16.triggered.connect(self.PlotR)
        self.actionOrbitRMSMachine.triggered.connect(self.PlotRMS)
        self.actionOrbitRMSModel.triggered.connect(self.PlotRMS)

    #        self.UIGenerateModel.clicked.connect(self.GenerateModel)
#        self.UIGenerateTraining.clicked.connect(self.GenerateTraining)
#        self.UILoadData.clicked.connect(self.LoadData)
#        self.UICheckTraining.clicked.connect(self.CheckData)

    def TrainModel(self):
        if not self.hasModel:
            self.OpenFile()
        if not self.hasModel:
                return

        nsam = int(str(self.UITrainSamples.text()))
        errorbx = float(str(self.UIOrbitX.text())) * 1e-3
        errangx = float(str(self.UIAngleX.text())) * 1e-3
        errorby = float(str(self.UIOrbitY.text())) * 1e-3
        errangy = float(str(self.UIAngleY.text())) * 1e-3
        errerg = float(str(self.UIEnergy.text()))
        errbpm = float(str(self.UIBPM.text())) * 1e-3
        # generate a training set
        self.model.generateTrainingsData(nsam, [errorbx, errangx, errorby, errangy, errerg, errbpm])
        nepoch = int(str(self.UITrainEpochs.text()))
        n = int(self.model.xdata.shape[0]*0.75)  # 75 percent are training and rest testing
        # run tf.keras
        self.train.runTF(self.model.xdata[0:n, :],
                        self.model.ydata[0:n, :],
                        self.model.xdata[n:, :],
                        self.model.ydata[n:, :], nepoch)
        self.hasTraining = True

    def OpenFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "DataMiner File",
                                                  "/sf/data/applications/BD-DataMiner/AramisOrbit/data",
                                                  "HDF5 Files (*.h5)", options=options)
        if fileName is None:
            return
        self.data.open(fileName)
        self.hasFile = True
        # generate madx model from machine settings in file
        self.model.updateModel(self.data.mag, self.data.energy)
        # prepare data from the machine
        bpm = [name.split('.MARK')[0].replace('.','-') for name in self.model.name]
        self.data.getBPM(bpm)
        self.hasModel = True

#------------------------------

    def CheckData(self):
        if not self.hasMachineData:
            self.LoadData()
        if not self.hasTraining:
            self.TrainModel()

        pred = self.train.predict(self.data.data)
        rms = np.std(pred[:, 0])*1e3
        self.UIOrbitX.setText('%f' % rms)
        rms = np.std(pred[:, 1])*1e3
        self.UIAngleX.setText('%f' % rms)
        rms = np.std(pred[:, 2])*1e3
        self.UIOrbitY.setText('%f' % rms)
        rms = np.std(pred[:, 3])*1e3
        self.UIAngleY.setText('%f' % rms)
        rms = np.std(pred[:, 4])
        self.UIEnergy.setText('%f' % rms)

        n=int(self.data.data.shape[1]/2)
        datacor=self.data.data-np.transpose(np.matmul(self.model.r,np.transpose(pred)))


        self.axes1.clear()
        self.axes1.plot(self.model.s, np.std(self.data.data[:, 0:n], axis=0))
        self.axes1.plot(self.model.s, np.std(self.data.data[:, n:], axis=0))

        self.axes2.clear()
        self.axes2.plot(self.model.s, np.std(datacor[:, 0:n], axis=0))
        self.axes2.plot(self.model.s, np.std(datacor[:, n:], axis=0))

        self.axes3.clear()
        self.axes4.clear()
        self.canvas.draw()




    def TrainModelOld(self):
        if not self.hasData:
            self.GenerateTraining()

        nepoch = int(str(self.UITrainEpochs.text()))
        n = int(self.model.xdata.shape[0]*0.75)  # 75 percent are training and rest testing
        x_train = self.model.xdata[0:n, :]
        y_train = self.model.ydata[0:n, :]
        x_test = self.model.xdata[n:, :]
        y_test = self.model.ydata[n:, :]

        y_predict, loss, acc, val_loss, val_acc = self.train.runTF(x_train, y_train, x_test, y_test, nepoch)
        ep = np.linspace(1, nepoch, num=nepoch)
        self.hasTraining = True
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
        errorbx = float(str(self.UIOrbitX.text()))*1e-3
        errangx = float(str(self.UIAngleX.text()))*1e-3
        errorby = float(str(self.UIOrbitY.text()))*1e-3
        errangy = float(str(self.UIAngleY.text()))*1e-3
        errerg = float(str(self.UIEnergy.text()))
        errbpm = float(str(self.UIBPM.text()))*1e-3
        fluc=self.model.generateTrainingsData(nsam,[errorbx,errangx,errorby,errangy,errerg,errbpm])
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


# plotting routine
    def PlotRMS(self):
        if not self.hasModel:
            return
        name = str(self.sender().objectName())
        x = self.model.s
        nx = len(x)
        if 'Machine' in name:
            y1 = np.std(self.data.data[:, 0:nx], axis=0)
            y2 = np.std(self.data.data[:, nx:], axis=0)
        if 'Model' in name:
            if not self.hasTraining:
                return
            y1 = np.std(self.model.xdata[:, 0:nx], axis=0)
            y2 = np.std(self.model.xdata[:, nx:], axis=0)
        self.axes.clear()
        self.axes.plot(x, y1, label=r'X')
        self.axes.plot(x, y2, label=r'Y')
        self.axes.set_xlabel(r'$s$ (m)')
        self.axes.set_ylabel(r'$\sigma_{x,y}$ (mm)')
        self.axes.legend()
        self.canvas.draw()

    def PlotR(self):
        if not self.hasModel:
            return
        name = str(self.sender().objectName())
        x = self.model.s
        y1 = self.model.rx5
        y2 = self.model.ry5
        l1 = r'$R_{16}$'
        l2 = r'$R_{36}$'
        ylab = r'$R_{16}, R_{36}$ (m)'
        if 'R11' in name:
            y1 = self.model.rx1
            y2 = self.model.ry3
            l1 = r'$R_{11}$'
            l2 = r'$R_{33}$'
            ylab = r'$R_{11}, R_{33}$'
        if 'R12' in name:
            y1 = self.model.rx2
            y2 = self.model.ry4
            l1 = r'$R_{12}$'
            l2 = r'$R_{34}$'
            ylab = r'$R_{12}, R_{34}$ (m)'

        self.axes.clear()
        self.axes.plot(x,y1,label=l1)
        self.axes.plot(x,y2,label=l2)
        self.axes.set_xlabel(r'$s$ (m)')
        self.axes.set_ylabel(ylab)
        self.axes.legend()
        self.canvas.draw()
# initializing matplotlib
    def initmpl(self):
        self.fig=Figure()
        self.axes=self.fig.add_subplot()
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

