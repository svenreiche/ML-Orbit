import os
os.environ["EPICS_CA_ADDR_LIST"]="sf-cagw"
os.environ["EPICS_CA_SERVER_PORT"]="5062"

import Model
import MLOrbit

#
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ML = MLOrbit.MLOrbit()
    ML.prepareData()
    ML.runTF()