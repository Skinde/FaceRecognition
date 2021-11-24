import cv2
import os
import face_recognition
import pandas as pd
import math
import random
from rtree import index
import matplotlib.pyplot as plt

currentDirectory = os.getcwd()
currentDirectory = currentDirectory + "/lfw"

p = index.Property()
p.dimension = 128
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index('rtree', properties=p)

def self_merge(list_inp):
    output = []
    for elem in list_inp:
        output.append(elem)
    for elem in list_inp:
        output.append(elem)
    return tuple(output)

def concatanate(list_inp1, list_inp2):
    output = []
    for elem in list_inp1:
        output.append(elem)
    for elem in list_inp2:
        output.append(elem)
    return tuple(output)

def euclidian_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise Exception("Euclidian Distance: Vectors are not same length")
    distancia = 0
    for i in range(len(vector1)):
        distancia = distancia + (vector1[i] - vector2[i])**2
    return math.sqrt(distancia)


def build_histogram(tests):
    history = []
    for i in range(tests):
        folders = os.listdir(currentDirectory)
        folder1 = folders[random.randint(0,len(folders) -1)]
        folder2 = folders[random.randint(0,len(folders) -1)]
        image1 = face_recognition.load_image_file(currentDirectory + "/" + folder1 + "/" + os.listdir(currentDirectory + "/" + folder1)[0])
        image2 = face_recognition.load_image_file(currentDirectory + "/" + folder2 + "/" + os.listdir(currentDirectory + "/" + folder2)[0])
        face_list1 = face_recognition.face_encodings(image1)
        face_list2 = face_recognition.face_encodings(image2)
        if len(face_list1) != 0 and len(face_list2) != 0:
            history.append(euclidian_distance(face_list1[0], face_list2[0]))
    plt.hist(history)
    plt.show()
    

def build_tree(test_size):
    i = 0
    for folder in os.listdir(currentDirectory):
        for image in os.listdir(currentDirectory + "/" + folder):
            image = face_recognition.load_image_file(currentDirectory + "/" + folder + "/" + image)
            face_list = face_recognition.face_encodings(image)
            if len(face_list) != 0:
                tuple_faces = face_list[0]
                vector = self_merge(tuple_faces)
                print("Loading " + folder +  " number: " + str(i))
                idx.insert(i, vector)
        i = i+1
        if (i==test_size):
            break
def knn(query):
    return(list(idx.intersection(self_merge(query))))

def range_search(query, r):
    query_decreased = list(query)
    query_increased = list(query)
    for i in range(len(query_increased)):
        query_increased[i] = query_increased[i] + r
    for i in range(len(query_decreased)):
        query_decreased[i] = query_decreased[i] - r
    return(list(idx.intersection(concatanate(query_decreased, query_increased))))

build_histogram(5000)
build_tree(2000)
image = face_recognition.load_image_file("/home/skinde/Documents/BD2LAB10/lfw/Donald_Rumsfeld/Donald_Rumsfeld_0001.jpg")
face_list = face_recognition.face_encodings(image)
print(knn(face_list[0]))
print(range_search(face_list[0], 0.2))