import h5py
import numpy as np
import matplotlib.pyplot as plt

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

    def PCA(self,loc):


        nshape = self.data.shape
        nbpm = nshape[1]
        nsam = nshape[0]
        self.loc = loc


        # step 0 - trying the SVD

#        X = self.data[:,0:nbpm]
#        Xcen = X - X.mean(axis=0)
#        U, s, Vt = np.linalg.svd(Xcen)
#        c1 = Vt.T[:,0]
#        c2 = Vt.T[:,1]
#        c3 = Vt.T[:,2]

#        plt.plot(c1)
#        plt.plot(c2)
#        plt.plot(c3)
#        plt.show()
#        plt.semilogy(s[0:10])
#        plt.show()

#        X = self.data[:, nbpm:]
#        Xcen = X - X.mean(axis=0)
#        U, s, Vt = np.linalg.svd(Xcen)
#        c1 = Vt.T[:, 0]
#        c2 = Vt.T[:, 1]
#        c3 = Vt.T[:, 2]

#        plt.plot(c1)
#        plt.plot(c2)
#        plt.plot(c3)
#        plt.show()
#        plt.semilogy(s[0:10])
#        plt.show()


        # step 1 - reference error sources
        nsrc = len(loc)

        self.ref = np.zeros((nsrc, nsam))

        for i in range(nsrc):   # taking out all correlation
            self.ref[i,:] = self.data[:, loc[i]]
            for j in range(i):
                slope, intercep, r, pval, stderr = stats.linregress(self.ref[j,:], self.ref[i,:])
                self.ref[i, :] -= slope * self.ref[j, :]

        # step 2 - calculate toe correlation function and the jitter budget
        self.r = np.zeros((nsrc,nbpm))

        # the jitter budgets
        self.jit = np.zeros((nsrc+1,nbpm))

        ires = []
        xres =[]
        for i in range(nbpm):
            dist = self.data[:, i]
            self.jit[0,i] = np.std(dist)
            for j in range(nsrc):
                if np.std(self.ref[j, :]) > 0:
                    slope, intercep, r, pval, stderr = stats.linregress(self.ref[j,:], dist)
                else:
                    slope = 0
                self.r[j,i] = slope
                dist = dist - slope * self.ref[j,:]
                self.jit[j+1,i] = np.std(dist)
            xres.append(dist[0:1000])



        #step 2 - do SVD on residual fluctuation
        X = np.transpose(np.array(xres))

        Xcen = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(Xcen)
        plt.plot(s[0:10])
        plt.show()
        c1 = np.abs(Vt.T[:, 0])
        c1 = c1 /np.max(c1) * np.max(self.jit[-1,:])
        plt.plot(c1)
        plt.show()

        plt.plot(self.jit[-1,:])
        plt.plot(c1)
        plt.show()

        for j in range(0,10):
            c2 = np.abs(Vt.T[:, j]*s[j])
            plt.plot(c2)
            plt.show()


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

