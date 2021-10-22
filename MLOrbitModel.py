import sys

import tempfile
from os import chdir, system
import matplotlib.pyplot as plt
# search path for online model
sys.path.append('/sf/bd/applications/OnlineModel/current')

import numpy as np
import scipy.optimize as sop
import OMMadxLat
import OMFacility


class OrbitModel:
    def __init__(self):
        # initialize the model
        self.SF = OMFacility.Facility()
        self.Madx = OMMadxLat.OMMadXLattice()
        # define the beamline section
        sec = self.SF.getSection('SARBD02')
        path = sec.mapping
        self.line = self.SF.BeamPath(path)
        self.line.setRange('S30CB13', 'SARBD01')

    def generateTrainingsData(self, Nsam, scale=[0.02, 0.002, 0.02, 0.002, 0.1, 0.001]):
        self.ydata = np.random.normal(0, 1, size=(Nsam, 5))
        for i in range(5):
            self.ydata[:, i] *= scale[i]
        self.xdata = np.transpose(np.matmul(self.r, np.transpose(self.ydata)))
        self.xdata += np.random.normal(0, scale[5], size=self.xdata.shape)

    def updateModel(self, maglist, energy):

        magFodo = []
        for key in maglist.keys():
            name = key.replace('-', '.')
            ele = self.SF.getElement(name)
            val = maglist[key]
            if '.MQ' in name:
                ele.k1 = val / ele.Length *1.
                if 'S30CB13-MQUA430' in key:  # save a few quads settings for a simple model
                    self.FodoQ1 = ele.k1
                if 'S30CB14-MQUA430' in key:
                    self.FodoQ2 = ele.k1
                if 'S30CB15-MQUA430' in key:
                    self.FodoQ3 = ele.k1
 #               print(ele.Name,ele.k1)
        self.SF.forceEnergyAt('SARCL02.MBND100', energy * 1e6)
        self.trackModel(energy)
 #       self.FODOModel()  # prepare matrices for the Fodo lattice

    def fitEvec(self,evec,N):
        Ne = evec.shape[1]
        self.corR=np.zeros((Ne,5))
        for i in range(Ne):
            self.corR[i,:] = self.fitRMatrix(evec[:,i],N)

    def fitRMatrix(self,evec,N):

        Nbpm=len(self.s)
        # prepare the eigenfunction
        self.revec = evec
        # initial guess
        x0 = np.array([1,0,1,0,1])
        x0[0] = self.revec[0]
        x0[2] = self.revec[Nbpm]
        x0[4] = -np.max(np.abs(self.revec))/np.max(np.abs(self.r[:,4]))
        res = sop.minimize(self.fitfun,x0,args=(N),method='Nelder-Mead',tol=0.01,options={'disp':False})
        return res.x


    def fitfun(self,x,N):

        NBPM=len(self.s)
        res = self.revec-x[0]*self.r[:,0]-x[4]*self.r[:,4]
        for i in range(1, 4):
            res -= x[i]*self.r[:,i]
        res1 = np.sum(res[0:N]**2)
        res2 = np.sum(res[NBPM:NBPM+N]**2)
        return res1+res2

    def trackModel(self, energy):
        self.writeLattice(energy)
        self.writeLatticeTracking()
        tempdir = tempfile.TemporaryDirectory()
        #        tempdir='.'
        with open(tempdir.name + '/tmp-lattice.madx', 'w') as f:
            for line in self.Madx.cc:
                f.write(line)
        print('Tracking with MadX...')
        chdir(tempdir.name)
        system('madx tmp-lattice.madx')
        res, self.name = self.parseOutput(tempdir.name)
        self.s = res[:, 0]
        self.rx1 = res[:, 1]
        self.rx2 = res[:, 2]
        self.rx3 = res[:, 3]
        self.rx4 = res[:, 4]
        self.rx5 = res[:, 5]
        self.ry1 = res[:, 6]
        self.ry2 = res[:, 7]
        self.ry3 = res[:, 8]
        self.ry4 = res[:, 9]
        self.ry5 = res[:, 10]
        r1 = np.concatenate((res[:, 1], res[:, 6]))
        r2 = np.concatenate((res[:, 2], res[:, 7]))
        r3 = np.concatenate((res[:, 3], res[:, 8]))
        r4 = np.concatenate((res[:, 4], res[:, 9]))
        r5 = np.concatenate((res[:, 5], res[:, 10]))
        self.r = np.transpose(np.stack([r1, r2, r3, r4, r5]))
        print('R-Matrix derived')
        tempdir.cleanup()

    def writeLattice(self, energy):
        self.Madx.clear()
        self.Madx.write('option,-echo;\n')
        self.Madx.write('betax0=30;\n')
        self.Madx.write('betay0=30;\n')
        self.Madx.write('alphax0=0;\n')
        self.Madx.write('alphay0=0;\n\n')
        self.Madx.write('beam, particle=electron,energy=%f,sigt=1e-3,sige=1e-4;\n\n' % energy)

        self.SF.setRegExpElement('S20SY02', 'MK.C0.0', 'cory', 0)
        self.SF.setRegExpElement('S20SY02', 'MBND', 'angle', 0)
        self.SF.setRegExpElement('S10DI01', 'MBND', 'angle', 0)
        # write the lattice
        self.line.writeLattice(self.Madx, self.SF)  # write lattice to madx

    def writeLatticeTracking(self):
        self.Madx.write('use, sequence=swissfel;\n')
        self.Madx.write('select, flag=twiss, column=NAME,S,RE11,RE12,RE13,RE14,RE16,RE31,RE32,RE33,RE34,RE36;\n')
        self.Madx.write(
            'twiss, range=s30cb13.dbpm420.mark/#e,rmatrix, sequence=swissfel,betx=betax0,bety=betay0,alfx=alphax0,alfy=alphay0,file="twiss.dat";\n')
        self.Madx.write('plot, haxis = s, vaxis = re11, re12, re33,re34, range =  #s/#e,colour=100;\n')
        self.Madx.write('plot, haxis = s, vaxis = re31, re32, re13,re14, range =  #s/#e,colour=100;\n')
        self.Madx.write('plot, haxis = s, vaxis = re16, re36, range =  #s/#e,colour=100;\n')
        self.Madx.write('exit;')

    def parseOutput(self, path):

        start = False
        skipline = False
        res = []
        name = []
        with open(path + '/twiss.dat') as f:
            for line in f:
                if '* NAME' in line:
                    start = True
                    skipline = True
                    continue
                elif start is False:
                    continue
                elif skipline is True:
                    skipline = False
                    continue
                else:
                    if 'BPM' in line and '.MARK' in line:
                        val = line.split()
                        res.append(np.array([float(x) for x in val[1:]]))
                        name.append(val[0][1:-1])
        return np.array(res), name


# not needed any more

    def FODOModel(self):
        Lcell = 9.1
        Lbpm = 0.1
        self.Lqua = 0.15
        z0BPM = 412.9760 + Lbpm * 0.5   # half BPM length to move to electric center
        z0Qua = 413.2500 - self.Lqua * 0.5  # half quad length to move to start of quad element
        L1 = z0Qua - z0BPM
        L2 = Lcell - L1 - self.Lqua
        self.D1 = np.array([[1, L1], [0, 1]])
        self.D2 = np.array([[1, L2], [0, 1]])

    def getResponse(self,scl0):
        scl=1+0.01*scl0
        return self.getR(scl*self.FodoQ1, scl*self.FodoQ2, scl*self.FodoQ3)

    def getR(self, q1, q2, q3):

        res = np.zeros((4, 4))
        Rx = np.array([[1, 0], [0, 1]])
        Ry = np.array([[1, 0], [0, 1]])
        res[0, 0] = Rx[0, 0]
        res[1, 0] = Rx[0, 1]
        res[2, 0] = Ry[0, 0]
        res[3, 0] = Ry[0, 1]
        idx = 1
        print('focal length',1/q1/self.Lqua)
        for q in [q1, q2, q3]:
            k = np.sqrt(np.abs(q))
            co = np.cos(k * self.Lqua)
            si = np.sin(k * self.Lqua)
            ch = np.cosh(k * self.Lqua)
            sh = np.sinh(k * self.Lqua)
            QF = np.squeeze(np.array([[co, si / k], [-si * k, co]]))
            QD = np.squeeze(np.array([[ch, sh / k], [sh * k, ch]]))

            if q > 0:
                Rx = np.matmul(self.D1, Rx)
                Rx = np.matmul(QF, Rx)
                Rx = np.matmul(self.D2, Rx)
                Ry = np.matmul(self.D1, Ry)
                Ry = np.matmul(QD, Ry)
                Ry = np.matmul(self.D2, Ry)
            else:
                Rx = np.matmul(self.D1, Rx)
                Rx = np.matmul(QD, Rx)
                Rx = np.matmul(self.D2, Rx)
                Ry = np.matmul(self.D1, Ry)
                Ry = np.matmul(QF, Ry)
                Ry = np.matmul(self.D2, Ry)
            res[0, idx] = Rx[0, 0]
            res[1, idx] = Rx[0, 1]
            res[2, idx] = Ry[0, 0]
            res[3, idx] = Ry[0, 1]
            idx += 1
        return res
