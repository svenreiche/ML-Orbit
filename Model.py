import sys
import tempfile
from os import chdir, system
# search path for online model
sys.path.append('/sf/bd/applications/OnlineModel/current')

import matplotlib.pyplot as plt
import numpy as np

import OMMadxLat
import OMFacility
import OMMachineInterface

class Model:
    def __init__(self):

        #initialize the model
        self.SF = OMFacility.Facility()
        self.SF.forceEnergyAt('SARCL02.MBND100', 3e9)
        self.MI = OMMachineInterface.MachineInterface()
        self.SF.writeFacility(self.MI)   # register the epics channels
        self.Madx = OMMadxLat.OMMadXLattice()
        # define the beamline section
        sec = self.SF.getSection('SARBD02')
        path = sec.mapping
        self.line = self.SF.BeamPath(path)
        self.line.setRange('SARCL01', 'SARBD01')



    def prepareData(self,Nsam,scale=[1,1,1,1,1]):
        self.ydata = np.random.normal(0, 1, size=(Nsam, 5))
        for i in range(5):
            self.ydata[:, i] *= scale[i]
        self.xdata = np.transpose(np.matmul(self.r,np.transpose(self.ydata)))


    def updateModelFromMachine(self):
        self.MI.updateIDs(self.SF)
        self.MI.updateMagnets(self.SF)

    def trackModel(self, variables={}):
        self.writeLattice()
        for var in variables.keys():
            self.Madx.write('%s := %s;\n' % (var, variables[var]))
        self.writeLatticeTracking()
        tempdir = tempfile.TemporaryDirectory()
        with open(tempdir.name + '/tmp-lattice.madx', 'w') as f:
            for line in self.Madx.cc:
                f.write(line)
        print('Tracking with MadX...')
        chdir(tempdir.name)
        system('madx tmp-lattice.madx')
        res, name = self.parseOutput(tempdir.name)
        self.s = res[:, 0]
        self.r1 = np.concatenate((res[:, 1], res[:, 6]))
        self.r2 = np.concatenate((res[:, 2], res[:, 7]))
        self.r3 = np.concatenate((res[:, 3], res[:, 8]))
        self.r4 = np.concatenate((res[:, 4], res[:, 9]))
        self.r5 = np.concatenate((res[:, 5], res[:, 10]))
        self.r = np.column_stack((self.r1, self.r2, self.r3, self.r4, self.r5))
        print('R-Matrix derived')
        tempdir.cleanup()

    def writeLattice(self):
        self.Madx.clear()
        self.Madx.write('option,-echo;\n')
        self.Madx.write('betax0=30;\n')
        self.Madx.write('betay0=30;\n')
        self.Madx.write('alphax0=0;\n')
        self.Madx.write('alphay0=0;\n\n')
        self.Madx.write('beam, particle=electron,energy=3000,sigt=1e-3,sige=1e-4;\n\n')

        self.SF.setRegExpElement('S20SY02', 'MK.C0.0', 'cory', 0)
        self.SF.setRegExpElement('S20SY02', 'MBND', 'angle', 0)
        self.SF.setRegExpElement('S10DI01', 'MBND', 'angle', 0)
        # write the lattice
        self.line.writeLattice(self.Madx, self.SF)  # write lattice to madx

    def writeLatticeTracking(self):
        self.Madx.write('use, sequence=swissfel;\n')
        self.Madx.write('select, flag=twiss, column=NAME,S,RE11,RE12,RE13,RE14,RE16,RE31,RE32,RE33,RE34,RE36;\n')
        self.Madx.write('twiss, range=#s/#e,rmatrix, sequence=swissfel,betx=betax0,bety=betay0,alfx=alphax0,alfy=alphay0,file="twiss.dat";\n')
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
        return np.array(res),name

    def plotRMatrix(self,res):
        plt.plot(res[:, 0], res[:, 1], label=r'$R_{11}$')
        plt.plot(res[:, 0], res[:, 2], label=r'$R_{12}$')
        plt.plot(res[:, 0], res[:, 8], label=r'$R_{33}$')
        plt.plot(res[:, 0], res[:, 9], label=r'$R_{34}$')
        plt.legend()
        plt.show()
        plt.plot(res[:, 0], res[:, 3], label=r'$R_{13}$')
        plt.plot(res[:, 0], res[:, 4], label=r'$R_{14}$')
        plt.plot(res[:, 0], res[:, 6], label=r'$R_{31}$')
        plt.plot(res[:, 0], res[:, 7], label=r'$R_{32}$')
        plt.legend()
        plt.show()
        plt.plot(res[:, 0], res[:, 5], label=r'$R_{16}$')
        plt.plot(res[:, 0], res[:, 10], label=r'$R_{36}$')

        plt.legend()
        plt.show()
