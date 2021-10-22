import h5py
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
from scipy import stats

class OrbitData:
    def __init__(self):
        self.file = None
        self.mag = {}
        self.energy = 5800  # in MeV
        self.nbpm = 0

    def open(self, file):
        if not self.file is None:
            self.file.close()
        self.file = h5py.File(file,'r')
        self.getMagnets()
        self.getEnergy()

    def getMagnets(self):
        self.mag.clear()
        for key in self.file['experiment'].keys():
            if '-MQ' in key:
                name = key.split(':')[0]
                self.mag[key] = self.file['experiment'][key]['K1L-SET'][()]

    def getEnergy(self):
        self.energy = self.file['experiment']['SARCL02-MBND100']['ENERGY-OP'][()]

    def getBPM(self,names):
        if self.file is None:
            return
        ny = len(names)
        nx = -1

        for iy, name in enumerate(names):
            if name in self.file['scan_1']['data'].keys():
 #               print('reading BPM:', name)
                dset = self.file['scan_1']['data'][name]['X1'][()]
                if nx < 0:
                    nx = len(dset)
                    self.data = np.zeros((nx, 2*ny))
                self.data[:, iy] = dset
                dset = self.file['scan_1']['data'][name]['Y1'][()]
                self.data[:, iy+ny] = dset
            else:
                print('Missing Dataset for', name)

            self.data[:,:] -= self.data[0,:]

        self.data -= self.data.mean(axis=0)   # take out center of mass

    def PCA(self):


        nshape = self.data.shape
        nbpm = nshape[1]
        nsam = nshape[0]

        # Step 1 - SVD

        self.datasave = copy.deepcopy(self.data)
        self.optx=np.zeros((10))
        self.opty=np.zeros((10))
        evecold = np.zeros((nbpm,2))
        Nt=4000

        X = self.data[:Nt,:]
        U, s, Vt = np.linalg.svd(X)
        self.svd = s**2/np.sum(s**2)
        self.evec = Vt.T

        # the jitter budgets
        self.nevec = 6
        self.jit = np.zeros((self.nevec+1,nbpm))

        self.r1 = np.zeros(nbpm)
        for i in range(self.nevec):
            self.r1 += self.svd[i]*self.evec[:,i]

        self.jitsrc=np.zeros((self.nevec,nsam))

        for isrc in range(self.nevec):
            self.jit[isrc, :] = np.std(self.data, axis=0)
            cvec = self.evec[:, isrc]
            self.jitsrc[isrc,:]=np.dot(self.data,cvec)
            for ibpm in range(nbpm):
                self.data[:,ibpm] -=self.jitsrc[isrc,:]*cvec[ibpm]
        self.jit[self.nevec, :] = np.std(self.data, axis=0)

        return




    def ReconstructR(self, model):

            s = model.s
            nbpm = len(s)

            ryc = np.zeros((2,nbpm))
            ayc = np.zeros((1))
            r34=model.ry4
            r33=model.ry3

            scl = np.polyfit(self.ry[1,0:6],r34[0:6],1)[0]
            ryc[1,:] = scl*self.ry[1,:]
            plt.plot(s,r34)
            plt.plot(s,self.ry[1,:]*scl)
            plt.title(r'$R_{34}$')
            plt.show()

            dr = self.ry[0,0:8]-r33[0:8]
            ayc[0] = np.polyfit(ryc[1,0:8],dr,1)[0]
            ryc[0,:] = self.ry[0,:] - ayc[0]*ryc[1,:]
            plt.plot(s,r33)
            plt.plot(s,ryc[0,:])
            plt.title(r'$R_{33}$')
            plt.show()

            # x - plane
            rxc = np.zeros((3,nbpm))

            r16=model.rx5
            r12=model.rx2
            r11=model.rx1

            scl = np.polyfit(self.rx[2,7:12],r16[7:12],1)[0]
            rxc[2,:] = scl*self.rx[2,:]
            plt.plot(s, r16)
            plt.plot(s,self.rx[2,:]*scl)
            plt.title(r'$R_{16}$')
            plt.show()

            scl = np.polyfit(self.rx[1, 0:7], r12[0:7], 1)[0]
            rxc[1, :] = scl * self.rx[1, :]
            dr = rxc[1,7:12]-r12[7:12]
            axc = np.polyfit(rxc[2,7:12],dr,1)[0]
            rxc[1, :] = rxc[1,:] - axc*rxc[2,:]
            plt.plot(s, r12)
            plt.plot(s, rxc[1,:])
            plt.title(r'$R_{12}$')
            plt.show()



            return

            dr = self.ry[0,0:6]-r33[0:6]
            ayc[0] = np.polyfit(ryc[1,0:6],dr,1)[0]
            ryc[0,:] = self.ry[0,:] - ayc[0]*ryc[1,:]
            plt.plot(s,r33)
            plt.plot(s,ryc[0,:])
            plt.show()

