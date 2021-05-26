import h5py

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


    def getBPMs(self):
        if self.file is None:
            return
        bpms=[]
        for key in self.file['BeamSynch'].keys():
            name = key.split(':')[0]
            if not name in bpms:
                bpms.append(name)

        print(bpms)


