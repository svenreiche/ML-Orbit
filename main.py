import os
import sys
os.environ["EPICS_CA_ADDR_LIST"]="sf-cagw"
os.environ["EPICS_CA_SERVER_PORT"]="5062"

import numpy as np
from scipy import stats
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


import matplotlib.pyplot as plt

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
        self.hasFile = False
        self.hasModel = False
#        self.hasTraining = False
#        self.hasPCA = False
# event
        self.actionLoad.triggered.connect(self.OpenFile)
        self.UIPCA.clicked.connect(self.AnalyseOrbit)
        self.UIRepPrev.clicked.connect(self.newReportPlot)
        self.UIRepNext.clicked.connect(self.newReportPlot)
        self.UIRepID.editingFinished.connect(self.newReportPlot)
#        self.UIRMatrix.clicked.connect(self.ReconstructRMatrix)
#        self.UITrainModel.clicked.connect(self.TrainModel)

# plotting
#        self.actionFlucX.triggered.connect(self.PlotFluc)
#        self.actionFlucY.triggered.connect(self.PlotFluc)
#        self.actionFlucRes.triggered.connect(self.PlotFluc)
#        self.actionPCACorX.triggered.connect(self.PlotCor)
#        self.actionPCACorY.triggered.connect(self.PlotCor)

#        self.UIPlotCor.clicked.connect(self.PlotCor)

#        self.actionValidX.triggered.connect(self.PlotValid)
#        self.actionValidXP.triggered.connect(self.PlotValid)
#        self.actionValidY.triggered.connect(self.PlotValid)
#        self.actionValidYP.triggered.connect(self.PlotValid)
#        self.actionValidEnergy.triggered.connect(self.PlotValid)
#        self.actionFlucContX.triggered.connect(self.PlotFlucCont)
#        self.actionFlucContY.triggered.connect(self.PlotFlucCont)
#        self.actionLoss.triggered.connect(self.PlotTFStat)
#        self.actionAccuracy.triggered.connect(self.PlotTFStat)


    #        self.UIGenerateModel.clicked.connect(self.GenerateModel)
#        self.UIGenerateTraining.clicked.connect(self.GenerateTraining)
#        self.UILoadData.clicked.connect(self.LoadData)
#        self.UICheckTraining.clicked.connect(self.CheckData)

    def OpenFile(self):
        """
        Reading in orbit data from measurements and populates the widgets
        """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "DataMiner File",
                                                  "/sf/data/measurements/2021/06/10",
                                                  "HDF5 Files (*.h5)", options=options)
        if fileName is None:
            return
        self.data.open(fileName)
        self.hasFile = True
        # generate madx model from machine settings in file
        self.model.updateModel(self.data.mag, self.data.energy)
        # prepare data from the machine
        self.bpm = [name.split('.MARK')[0].replace('.', '-') for name in self.model.name]

        self.data.getBPM(self.bpm)
        self.hasModel = True

    def AnalyseOrbit(self):

        if self.hasModel == False:
            return
        nbpm = len(self.model.s)

        self.data.PCA()
#        self.axes.clear()
#        self.axes.loglog(self.data.optx,self.data.opty)
#        self.axes.set_xlabel(r'# Samples')
#        self.axes.set_ylabel(r'$\Delta ||\vec{n}_j-\vec{n}_{j-1}||$')
#        self.axes.set_ylabel(r'$t$ (s)')
#        self.axes.set_title(r'Execution Time of PCA Algorithm')
#        self.canvas.draw()

        self.model.fitEvec(self.data.evec, 12)
        self.hasPCA = True
        self.PlotReport(0)
        return

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

    def newReportPlot(self):
        name = self.sender().objectName()
        if 'Prev' in name:
            inc = -1
        elif 'Next' in name:
            inc = 1
        else:
            inc = 0
        id = int(str(self.UIRepID.text())) + inc
        self.PlotReport(id)

    def PlotReport(self,irep):
        if self.hasPCA == False:
            return
        maxplot = 14+38
        if irep > maxplot:
            irep = 0
        if irep < 0:
            irep = maxplot
        haslegend = True
        self.UIRepID.setText('%d' % irep)

        nbpm = len(self.model.s)
        self.axes.clear()
        if irep == 0:
            self.axes.set_aspect('equal')
        else:
            self.axes.set_aspect('auto')
            self.axes.set_frame_on(True)
        if irep == 0:
            title ='Distribution of Principle Jitter Sources'
            data = self.data.svd[0:8]
            data[7]=np.sum(self.data.svd[7:])
            data *= 100.
            explode = (0,0,0,0,0.15,0.3,0.45,0.6)
            wedges, texts, autotexts =self.axes.pie(data,explode=explode,autopct='%1.1f%%',startangle=180)
            xlab=''
            ylab=''
            haslegend = False
        elif irep == 1 or irep == 2:
            if irep == 1:
                title = 'Relative Fluctuation in X'
            else:
                title = 'Relative Fluctuation in Y'
            xlab = r'BPM'
            ylab = r'$\sigma_{x,y}$ (rel)'
            if irep == 1:
                flucnorm= self.data.jit[0,0:nbpm]**2
            else:
                flucnorm = self.data.jit[0,nbpm:]**2
            botfluc = flucnorm*0
            x = np.arange(nbpm)
            for j in range(1, self.data.nevec+1):
                if irep == 1:
                    fluc = -(self.data.jit[j, 0:nbpm]**2-self.data.jit[j-1, 0:nbpm]**2)/flucnorm
                else:
                    fluc = -(self.data.jit[j, nbpm:]**2-self.data.jit[j-1, nbpm:]**2)/flucnorm
                self.axes.bar(x, fluc, bottom=botfluc, label=('%d. Component' % j))
                botfluc = botfluc + fluc
            if irep == 1:
                fluc = (self.data.jit[-1, 0:nbpm]**2)/flucnorm
            else:
                fluc = (self.data.jit[-1, nbpm:]**2)/flucnorm
            self.axes.bar(x, fluc, bottom=botfluc, label='Residual')
        elif irep == 3 or irep == 4:
            if irep == 3:
                title = 'Fluctuation in X'
                ylab = r'$\sigma_{x}$ ($\mu$m)'

            else:
                title = 'Fluctuation in Y'
                ylab = r'$\sigma_{y}$ ($\mu$m)'

            xlab = r'$z$ (m)'
            labels=['%d. Component' % j for j in range(nbpm)]
            labels[0]='Full'
            for j in range(0, self.data.nevec+1):
                if irep == 3:
                    fluc = self.data.jit[j, 0:nbpm]
                else:
                    fluc = self.data.jit[j, nbpm:]
                self.axes.plot(self.model.s, fluc*1e3, label=labels[j])

        elif irep > 4 and irep < 11:
            title = 'Eigenvector for Mode %d' % (irep-4)
            xlab = r'$z$ (m)'
            ylab = r'$\vec{v}$'
            rfit=0
            for i in range(5):
                rfit += self.model.corR[irep-5, i]*self.model.r[:, i]
            self.axes.plot(self.model.s, self.data.evec[0:nbpm, irep - 5] * self.data.svd[irep - 5], 'b', label='X')
            self.axes.plot(self.model.s, self.data.evec[nbpm:, irep - 5] * self.data.svd[irep - 5], 'r', label='Y')
            self.axes.plot(self.model.s, rfit[0:nbpm] * self.data.svd[irep - 5], 'b--', label='X-Fit')
            self.axes.plot(self.model.s, rfit[nbpm:] * self.data.svd[irep - 5], 'r--', label='Y-Fit')
        elif irep > 10 and irep < 15:
            haslegend = False
            idx = [[0, 1], [0, 2], [2, 3], [0, 4]]
            isel = irep-11
            ixy = idx[isel]
            ix = ixy[0]
            iy = ixy[1]
            lab = [r'$x$ ($\mu$m)',r'$x^\prime$ ($\mu$rad)',r'$y$ ($\mu$m)',r'$y^\prime$ ($\mu$rad)',r'$\Delta E/E$ ($10^{-6}$)']
            tit = [r'$X$',r'$X^\prime$',r'$Y$',r'$Y^\prime$',r'$\Delta E/E$']
            xlab = lab[ix]
            ylab = lab[iy]
            title = r'Phase Space Reconstruction for %s - %s' % (tit[ix], tit[iy])
            xdist = 0
            ydist = 0
            for ivec in range(6):
                xdist += self.data.jitsrc[ivec, :] * self.model.corR[ivec, ix]
                ydist += self.data.jitsrc[ivec, :] * self.model.corR[ivec, iy]
            fit=np.polyfit(xdist,ydist,1)
            ytmp=ydist-xdist*fit[0]
            xstd=np.std(xdist)
            ystd=np.std(ydist)
            zstd=np.std(ytmp)
            gamma=self.data.energy[0]/0.511
            emit = xstd*zstd*1e3*gamma
            if isel == 0 or isel == 2:
                title += r' (Jitter Emittance $\epsilon_n$ = %2.1f nm)' % emit
            elif isel == 3:
                title += r' (Energy Jitter $\Delta E/E$ = %3.3f %%)' % (ystd/10)
            self.axes.scatter(xdist*1e3, ydist*1e3, s=0.5)

        elif irep > 14 and irep < 14+nbpm:
            haslegend = False
            title = 'Measured Orbit at %s' % self.bpm[irep-15]
            xlab =r'$x$ ($\mu$m)'
            ylab =r'$y$ ($\mu$m)'
            self.axes.scatter(self.data.datasave[:,irep-15]*1e3,self.data.datasave[:,irep-15+nbpm]*1e3,s=0.5)
        else:
            return
        if haslegend:
            self.axes.legend()
        self.axes.set_xlabel(xlab)
        self.axes.set_ylabel(ylab)
        self.axes.set_title(title)
        self.canvas.draw()
        return

    def PlotCor(self):
        if self.hasPCA == False:
            return
        nbpm = len(self.model.s)
        self.axes.clear()
        sels = self.UIXPCA.selectedItems()
        for sel in sels:
            source = str(sel.text())
            idx = self.UIXPCA.row(sel)
            self.axes.plot(self.model.s,self.data.r[idx,0:nbpm],label='X vs %s' % source)
            self.axes.plot(self.model.s,self.data.r[idx,nbpm:],label='Y vs %s' % source)

        self.axes.legend()
        self.axes.set_ylabel(r'$<z_j\cdot z_{BPM}>$')
        self.axes.set_xlabel(r'$s$ (m)')
        self.canvas.draw()

#        if 'X' in name:
#            data = self.data.r[:,0:nbpm]
#        else:
#            data = self.data.r[:,nbpm:]

        #        ylab = r'$Cor(\vec{X},x_j)$'
#        leg = []
#            for i in range(data.shape[0]):
#                leg.append(self.bpm[self.data.locx[i]])
#        elif 'Y' in name:
#            data = self.data.ry
#            ylab = r'$Cor(\vec{Y},y_j)$'
#            leg = []
#            for i in range(0, data.shape[0]):
#                leg.append(self.bpm[self.data.locy[i] - len(self.bpm)])
#        self.axes.clear()
 #       for i in range(data.shape[0]):
 #           self.axes.plot(self.model.s,data[i,:])

#        self.axes.legend()
 #       self.axes.set_xlabel(r'$s$ (m)')
#        self.axes.set_ylabel(ylab)
 #       self.canvas.draw()





    def test(self):


        dist = self.data.data[:, 0]
        idx = np.argmin(np.squeeze(np.abs(dist[1:])))
        print(idx,dist[idx],self.data.data[idx,0])
        for i in range(4):
            xorb[i] = self.data.data[idx,i]


        plt.scatter(np.abs(dist),self.data.data[:,1],s=0.5)
        plt.xlim([0,0.001])
        plt.show()
        xref1 = res[0,:]*xorb[0]



        dif = (xorb-xref1)
 #       difref = res[1,:
        plt.plot(xorb)
        plt.plot(xref1)
        plt.show()
        plt.plot(dif)
 #       plt.plot(difref)
        plt.show()

    def tmp(self):

        nbpm = len(self.model.s)

        res = self.model.getResponse(0)
        xref = self.data.data[:,0]
        yref = self.data.data[:,0+nbpm]

        rms = np.zeros((2,3))
        for i in range(1,4):


            distx = self.data.data[:,i]
            disty = self.data.data[:,i+nbpm]

            if i == 2:
                distx*=0.9
                disty*=0.9

            distx -= res[0,i]*xref
            disty -= res[2,i]*yref

            distx /= res[1,i]
            disty /= res[3,i]

            alphax = np.polyfit(xref, distx, 1)[0]
            alphay = np.polyfit(yref, disty, 1)[0]

            distx -= alphax * xref
            disty -= alphay * yref

            rms[0,i-1] = np.std(distx)
            rms[1,i-1] = np.std(disty)


        plt.plot(rms[0,:])
        plt.plot(rms[1,:])

        plt.show()


    def AnalyseOrbitOLd(self):

        # step 1 - derive phase space distribution from the three BPM readings SARMA01-DBPM010/020/40
        noff = len(self.model.s)
        idx=-1
        L = 2.049 # distance between two quads
        for i,name in enumerate(self.model.name):
            if 'SARMA02.DBPM010.MARK' == name:
                idx = i
        if idx < 0:
                print('BPM not found')
                return
        xref = self.data.data[:, idx]   # use as reference distribution
        xpref = (self.data.data[:, idx+2]-xref)/2/L
        yref = self.data.data[:, idx+noff]
        ypref = (self.data.data[:, idx+2+noff]-yref)/2/L
        fitx = np.polyfit(xref,xpref,1)[0]
        fity = np.polyfit(yref,ypref,1)[0]
        xpref -= fitx * xref    # remove any correlation to have independent jitters
        ypref -= fity * yref
        xstd = np.std(xref)
        xpstd = np.std(xpref)
        ystd = np.std(yref)
        ypstd = np.std(ypref)

        # step 2 - correlation along the machine
        flucx= np.zeros((noff-idx))
        flucy= np.zeros((noff-idx))

        rmat = np.zeros((4,4,noff-idx))
        s = self.model.s[idx:]
        for i in range(idx,noff):
            flucx[i-idx] = np.std(self.data.data[:,i])
            flucy[i-idx] = np.std(self.data.data[:,i+noff])
            rmat[0, 0, i - idx] = np.polyfit(xref, self.data.data[:,i],1)[0]
            rmat[0, 1, i - idx] = np.polyfit(xpref, self.data.data[:,i],1)[0]
            rmat[2, 2, i - idx] = np.polyfit(yref, self.data.data[:, i+noff], 1)[0]
            rmat[2, 3, i - idx] = np.polyfit(ypref, self.data.data[:, i+noff], 1)[0]
            rmat[0, 2, i - idx] = np.polyfit(yref, self.data.data[:,i],1)[0]
            rmat[0, 3, i - idx] = np.polyfit(ypref, self.data.data[:,i],1)[0]
            rmat[2, 0, i - idx] = np.polyfit(xref, self.data.data[:, i+noff], 1)[0]
            rmat[2, 1, i - idx] = np.polyfit(xpref, self.data.data[:, i+noff], 1)[0]


        flucxcor = np.sqrt((rmat[0,0,:] * xstd)**2 + (rmat[0,1,:] * xpstd)**2 + (rmat[0,2,:] * ystd)**2 + (rmat[0,3,:] * ypstd)**2)
        flucycor = np.sqrt((rmat[2,0,:] * xstd)**2 + (rmat[2,1,:] * xpstd)**2 + (rmat[2,2,:] * ystd)**2 + (rmat[2,3,:] * ypstd)**2)


        plt.scatter(xref,yref,s=0.5)
        plt.show()

        plt.plot(s, flucx, label='Measured')
        plt.plot(s, flucxcor, label='Reconstructed')
        plt.xlabel('s (m)')
        plt.ylabel(r'$\sigma_x$ (mm)')
        plt.legend()
        plt.show()

        plt.plot(s, flucy)
        plt.plot(s, flucycor)
        plt.xlabel('s (m)')
        plt.ylabel(r'$\sigma_y$ (mm)')
        plt.legend()
        plt.show()

        plt.plot(s,rmat[0,0,:], label=r'$R_{11}$')
        plt.plot(s,rmat[2,2,:], label=r'$R_{33}$')
        plt.xlabel('s (m)')
        plt.ylabel(r'$R_{11}, R_{33}$')
        plt.legend()
        plt.show()


        plt.plot(s,rmat[0,1],label=r'$R_{12}$')
        plt.plot(s,rmat[2,3],label=r'$R_{34}$')
        plt.xlabel('s (m)')
        plt.ylabel(r'$R_{12}, R_{34}$ (m)')
        plt.legend()
        plt.show()

        plt.plot(s, rmat[0, 2], label=r'$R_{13}$')
        plt.plot(s, rmat[2, 0], label=r'$R_{31}$')
        plt.xlabel('s (m)')
        plt.ylabel(r'$R_{13}, R_{31}$ (m)')
        plt.legend()
        plt.show()

        plt.plot(s, rmat[0, 3], label=r'$R_{14}$')
        plt.plot(s, rmat[2, 1], label=r'$R_{32}$')
        plt.xlabel('s (m)')
        plt.ylabel(r'$R_{14}, R_{32}$ (m)')
        plt.legend()
        plt.show()

        #       plt.scatter(xref, xpref, s=0.5, label='X-Plane')
#       plt.scatter(yref, ypref, s=0.5, label='Y-Plane')
#       plt.legend()
#       plt.show()

        if True:
            return

        nz=5    # number of bpms

        xref = self.data.data[:, 0]
        xpref= xref*0
        xsig = np.zeros(nz)
        xsig[0] = np.std(xref)
        cor11 = np.zeros(nz)
        cor12 = np.zeros(nz)
        cor11[0] = 1.

        for i in range(1,nz):
            xdist = self.data.data[:,i]
            xsig[i] = np.std(xdist)
            cor11[i] = np.polyfit(xref, xdist, 1)[0]
            xdist = xdist - cor11[i]*xref
            if i == 1:
                xpref = xdist
                cor12[i] = 1
            else:
                cor12[i] = np.polyfit(xpref, xdist, 1)[0]
        s = self.model.s[0:nz]
        rx = self.model.rx1[0:nz]
        rxp = self.model.rx2[0:nz]

        cor12=cor12*rxp[1]/cor12[1]
        dif = rx[1]-cor11[1]
        corr = cor12*dif/cor12[1]
        plt.plot(s,cor11)
        plt.plot(s,rx)
        plt.plot(s,cor11+corr)
        plt.show()
        plt.plot(s,cor12)
        plt.plot(s,rxp)
        plt.show()

        if True:
                return



        ry = self.model.ry3[0:nz]
        ryp = self.model.ry4[0:nz]
        rxp /= rxp[1]
        ryp /= ryp[1]
        rd = self.model.rx5[0:nz]

        rmat=np.zeros((nz,5))     # the calculated rmatirx
        fluc = np.zeros((nz,2))    # the current fluctuation
        fluccor = np.zeros((nz,2))  # the corrected fluctuation
        flucres = np.zeros((nz,2))  # the residual fluctuation
        # first step: Jitter in x and y

        xref = self.data.data[:, 0]
        yref = self.data.data[:, noff]
        rmat[0, 0] = 1
        rmat[0, 2] = 1
        fluc[0, 0] = np.std(xref)
        fluc[0, 1] = np.std(yref)
        fluccor[0,0] = 0*fluc[0, 0]
        fluccor[0,1] = 0*fluc[0, 1]
        flucres[0,0] = 0
        flucres[0,1] = 0
        for i in range(1,nz):
            xdist = self.data.data[:, i]
            ydist = self.data.data[:, i + noff]
            fluc[i, 0] = np.std(xdist)
            fluc[i, 1] = np.std(ydist)
            corx = np.polyfit(xref, xdist, 1)[0]
            cory = np.polyfit(yref, ydist, 1)[0]
            rmat[i, 0] = corx
            rmat[i, 2] = cory
#            xdist = xdist - corx * xref
#            ydist = ydist - cory * yref
            fluccor[i, 0] = np.sqrt(fluc[i, 0]**2 - np.abs(corx)**2 * fluc[0, 0]**2)
            fluccor[i, 1] = np.sqrt(fluc[i, 1]**2 - np.abs(cory)**2 * fluc[0, 1]**2)
            if i == 1:
                xpref = xdist - corx * xref
                ypref = ydist - cory * yref
                rmat[i, 1] = 1
                rmat[i, 3] = 1
                flucres[i, 0] = 0
                flucres[i, 1] = 0
            else:
                corx = np.polyfit(xpref, xdist, 1)[0]
                cory = np.polyfit(ypref, ydist, 1)[0]
                rmat[i, 1] = corx
                rmat[i, 3] = cory
                flucres[i, 0] = np.sqrt(fluccor[i, 0] ** 2 - np.abs(corx) ** 2 * fluccor[1, 0] ** 2)
                flucres[i, 1] = np.sqrt(fluccor[i, 1] ** 2 - np.abs(cory) ** 2 * fluccor[1, 1] ** 2)


#            if i == 2:
#                plt.scatter(xref,self.data.data[:, i],s=0.5)
#                plt.scatter(xref,xdist,s=0.5)
#                plt.show()
#                plt.scatter(xpref,xdist,s=0.5)
#                plt.scatter(xpref,xdist-corx*xpref,s=0.5)
#                plt.show()


        plt.plot(s,rmat[:,0])
        plt.plot(s, rx)
        plt.show()

        plt.plot(s,rmat[:,2])
        plt.plot(s,ry)
        plt.show()

        plt.plot(s,rmat[:,1])
        plt.plot(s, rxp)
        plt.show()

        plt.plot(s,rmat[:,3])
        plt.plot(s, ryp)
        plt.show()


        plt.plot(s,rd)
        plt.show()

        plt.plot(s,fluc[:,0])
        plt.plot(s,fluccor[:,0])
        plt.plot(s,flucres[:,0])
        plt.show()

        plt.plot(s,fluc[:,1])
        plt.plot(s,fluccor[:,1])
        plt.plot(s,flucres[:,1])

        plt.show()
        return

    def ooo(self):
        rx=self.model.r[0:5,0]
        ry=self.model.r[0:5,1]
        x=self.data.data[12,0:5]

        a00=np.sum(rx*rx)
        a01=np.sum(rx*ry)
        a10=np.sum(rx*ry)
        a11=np.sum(ry*ry)

        detA = a00*a11-a01*a10
        c0=np.sum(rx*x)
        c1=np.sum(ry*x)

        B = (a00*c1-a10*c0)/detA/2
        A = (a11*c0-a01*c1)/detA/2


        xpre = rx*A+ry*B
        plt.plot(x)
        plt.show()
        plt.plot(rx)
#        plt.plot(ry*B)
#        plt.plot(rx*A+ry*B)
        plt.show()
        plt.plot(ry)
        plt.show()

    def tmp(self):
        x_data = self.train.model.predict(self.data.data)    # this is the prediction
        orb_model = np.transpose(np.matmul(self.model.r,np.transpose(x_data)))



        nepoch = int(str(self.UITrainEpochs.text()))
        x_test = np.zeros((5, 5))
        for i in range(5):
            x_test[i, i] = 1

        r_cor=np.transpose(self.train.runTFInv(x_data, y_data, x_test, nepoch))
        print(r_cor.shape)

#        for i in range(5):
#            plt.plot(self.model.r[:,i])
#            plt.plot(r_cor[:, i])
#            plt.show()


 #       rms = np.std(pred[:, 0])*1e3
 #       self.UIOrbitX.setText('%f' % rms)
 #       rms = np.std(pred[:, 1])*1e3
 #       self.UIAngleX.setText('%f' % rms)
 #       rms = np.std(pred[:, 2])*1e3
 #       self.UIOrbitY.setText('%f' % rms)
 #       rms = np.std(pred[:, 3])*1e3
 #       self.UIAngleY.setText('%f' % rms)
 #       rms = np.std(pred[:, 4])
 #       self.UIEnergy.setText('%f' % rms)







#-----------------------------------------------
# plotting routine

    def PlotTFStat(self):
        if not self.hasTraining:
            return
        name = str(self.sender().objectName())
        if 'Loss' in name:
            y1 = self.train.history.history['loss']
            y2 = self.train.history.history['val_loss']
            ylab = 'Loss Function'
        elif 'Accuracy' in name:
            y1 = self.train.history.history['accuracy']
            y2 = self.train.history.history['val_accuracy']
            ylab = 'Accuracy'
        else:
            return
        self.axes.clear()
        eph = self.train.history.epoch
        self.axes.plot(eph, y1, label='Training Set')
        self.axes.plot(eph, y2, label='Validation Set')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel(ylab)
        self.legend()
        self.canvas.draw()


    def PlotFlucCont(self):
        if not self.hasTraining:
            return
        name = str(self.sender().objectName())
        nx = len(self.model.s)
        y = self.model.ydata

        if 'X' in name:
            r = self.model.r[0:nx,:]
            lab = r'$\sigma_x$ (mm)'
        else:
            r = self.model.r[nx:,:]
            lab = r'$\sigma_y$'

        labels = ['X', 'XP', 'Y', 'YP', 'Energy']

        self.axes.clear()
        yold=self.model.s*0
        for i in range(5):
            orb = np.abs(r[:, i])*np.std(y[:, i])
            ynew = np.sqrt(yold**2+orb**2)
            self.axes.fill_between(self.model.s, ynew, yold,label=labels[i])
            yold = ynew
        self.axes.set_xlabel(r'$s$ (m)')
        self.axes.set_ylabel(lab)
        self.axes.legend()
        self.canvas.draw()


    def PlotValid(self):
        if not self.hasTraining:
            return
        name = str(self.sender().objectName())
        idx = 4
        title = 'Energy'
        if 'XP' in name:
            idx = 1
            title = 'XP'
        elif 'X' in name:
            idx = 0
            title = 'X'
        elif 'YP' in name:
            idx = 3
            title = 'YP'
        elif 'Y' in name:
            idx = 2
            title = 'Y'
        self.axes.clear()
        x = np.concatenate([y for x, y in self.train.val_dataset], axis=0)
        y = np.concatenate([y for y in self.train.predict], axis=0)
        y = np.reshape(y,x.shape)
        self.axes.scatter(x[:,idx],y[:,idx],s=0.5)
        self.axes.set_xlabel('Input Jitter')
        self.axes.set_ylabel('Reconstructed Jitter')
        self.axes.set_title('Jitter in %s' % title)
        self.canvas.draw()


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

