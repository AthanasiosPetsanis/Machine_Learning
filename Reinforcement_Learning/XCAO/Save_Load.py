'''
 =======================================================================================
 This function acts as the save_load module of the program!
 =======================================================================================
'''

import json,codecs
import cloudpickle
import pickle

import scipy.io as matlab
import numpy as np


def sensor_data(folder, filename):
    with open(folder + filename, 'rb') as handle:

        data1=json.load(handle)

        if 'contextElements' in data1.keys():
            attributes = data1["contextElements"]

            for x in attributes:
                z=x["attributes"]

                for x in z:
                    z = x['name']
                    if z == "Temperature":
                        temp = (x["value"])
                        temp = temp.split(",")
                        temp = int(temp[0]) + 0.1 * int(temp[1])

                    if z == "Humidity":
                        hum = float(x["value"])

                    if z == "CO2":
                        co2 = float(x["value"])

                    if z == "EnergyConsumption":
                        energy = (x["value"])
                        energy = energy.split(",")
                        energy = int(energy[0]) + 0.1 * int(energy[1])

                    if z == "EnergyPhotovoltaic":
                        Photovoltaic = (x["value"])
                        Photovoltaic = Photovoltaic.split(",")
                        Photovoltaic = int(Photovoltaic[0]) + 0.1 * int(Photovoltaic[1])

            data = [temp, hum, co2, energy, Photovoltaic]


        else:
            attributes=data1["attributes"]

            for x in attributes:

                z=x['name']
                if z=="Temperature":
                    temp=(x["value"])
                    temp=temp.split(",")
                    temp=int(temp[0])+0.1*int(temp[1])

                if z == "Humidity":
                    hum = float(x["value"])


                if z == "CO2":
                    co2 = float(x["value"])

                if z == "Consumption energy ":
                    energy = (x["value"])
                    energy= energy.split(",")
                    energy = int(energy[0]) + 0.1 * int(energy[1])

                if z == "Photovoltaic energy":
                    Photovoltaic = (x["value"])
                    Photovoltaic = Photovoltaic.split(",")
                    Photovoltaic = int(Photovoltaic[0]) + 0.1 * int(Photovoltaic[1])

                if z=="Ventilator":
                    Ventilator = (x["value"])


            data = [temp, hum, co2, energy, Photovoltaic, Ventilator]

        if 'updateAction' in data1.keys():
                updateAction = data1["updateAction"]








    return data








def save (data, folder, filename,module):

    if module=='mat':

        matlab.savemat(folder + filename,mdict={'exon': data},appendmat=True)

    else:
        with open (folder + filename,'wb') as handle:
          if module == 'pickle':
              if filename=='Symbolic_functions':

                  cloudpickle.dump(data,handle)

              else:

                  data={'exon': data}
                  pickle.dump(data ,handle)

                  handle.close()



          elif module == 'json':
              json.dump(data.tolist(), codecs.open(folder + filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,indent=4)  ### this saves the array in .json format







def load (folder, filename,module):        

    if module == 'mat':


        x = matlab.loadmat(folder + filename+'.mat',mdict=None, appendmat=True,byte_order=None,mat_dtype=True)
        data= x['exon']

    else:
        with open (folder + filename,'rb') as handle:
          if module == 'pickle':
              if filename=='Symbolic_functions':

                  data = cloudpickle.load(handle)

              else:


                  data = pickle.load(handle)
                  data = data['exon']

                  handle.close()
          elif module == 'json':
            data = json.load(handle)
            data= np.array(data)

    return data