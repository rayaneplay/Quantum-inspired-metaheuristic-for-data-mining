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

def binary_to_index(binary_array):
    index_array = []
    for i, bit in enumerate(binary_array, start=1):
        if bit == 1:
            index_array.append(i)
    return index_array

def get_values(binary_array, input_dict):
    values = []
    count = 0
    for i, bit in enumerate(binary_array, start=1):
        if bit == 1:
            values.append(input_dict[i])
            count += 1
    return values

def distance_manhattan(centroid, datapoint):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    dist = np.sum(np.abs(centroid - datapoint))
    return dist

def distance_minkowski(centroid, datapoint, p):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    dist = np.sum(np.abs(centroid - datapoint) ** p)
    return (dist ** (1/p))

def distance_euclidean(centroid, datapoint):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    dist = np.sqrt(np.sum((centroid - datapoint)**2))
    return dist

def distance_cosine(centroid, datapoint):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    up = np.sum(centroid * datapoint)
    down1 = np.sum(centroid * centroid)
    down2 = np.sum(datapoint * datapoint)
    dist = 1 - (up / (np.sqrt(down1) * np.sqrt(down2)))
    return dist

def distance_hamming(centroid, datapoint):
    centroid = np.array(centroid)
    datapoint = np.array(datapoint)
    dist =0
    for c,d in zip(centroid, datapoint):
        if c != d:
            dist += 1
    return dist

