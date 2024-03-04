import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt 
import random


def LoadDataSet(path , arg):
    with open (path,arg) as file :
        data=[]
        for line in file:
            data.append(line.split(','))
    dataset = np.array(data)
    return dataset

def GetInfo(dataset):
    print("Information about dataset")
    print ("Number of lines " + str(len(dataset)) )
    print ("Number of attributs " + str(len(dataset[0])) )
    print ("Number of elements in all items " +str(np.size(dataset)))


def preprocessing(dataset):
    
    converted_data =  [[float(element) for element in sublist[:-1]] for sublist in dataset]

    return (converted_data)


def list_to_dict(data):
    data_dict = {}
    for i, sublist in enumerate(data, start=1):
        data_dict[i] = sublist
    return data_dict

def elephant_to_dict(binary_array, input_dict):
    filtered_dict = {}
    for i, bit in enumerate(binary_array, start=1):
        if bit == 1:
            filtered_dict[i] = input_dict[i]
    return filtered_dict



