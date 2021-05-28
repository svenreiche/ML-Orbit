import h5py
import numpy as np

class OrbitData:
    def __init__(self):
        self.file = None
        self.mag = {}
        self.energy = 5800  # in MeV

    def open(self, file):
        if not self.file is None:
            self.file.close()
        self.file = h5py.File(file,'r')
        self.getMagnets()
        self.getEnergy()

 #       self.getBPMs()

    def getMagnets(self):
        self.mag.clear()
        for key in self.file['Epics'].keys():
            if '-M' in key and 'K1L-SET':
                name = key.split(':')[0]
                self.mag[name] = self.file['Epics'][key][()]

    def getEnergy(self):
        self.energy = self.file['Epics']['SARCL02-MBND100:ENERGY-OP'][()]


    def getBPM(self,names):
        if self.file is None:
            return
        ny = len(names)
        nx = -1
        refx = None
        refy = None

        for iy, name in enumerate(names):
            PV = name+':X1'
            if PV in self.file['BeamSynch'].keys():
                dset = self.file['BeamSynch'][PV][()]
                if nx < 0:
                    nx = len(dset)
                    self.data = np.zeros((nx, 2*ny))
                self.data[:, iy] = dset
            else:
                print('Missing Dataset for', PV)
            PV = name + ':Y1'
            if PV in self.file['BeamSynch'].keys():
                dset = self.file['BeamSynch'][PV][()]
                self.data[:, iy+ny] = dset
            else:
                print('Missing Dataset for', PV)

            self.data[:,:] -= self.data[0,:]





