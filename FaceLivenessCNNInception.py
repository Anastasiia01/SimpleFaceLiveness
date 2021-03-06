import tensorflow as tf
import sys
import argparse 
from ReadData import ReadData
from CustomModelCheckPoint import CustomModelCheckpoint
from CNNModel import CNNModel
from tensorflow.keras import datasets, layers, models
from keras.optimizers import SGD
from InceptionV4Model import InceptionV4Model
import numpy as np
import json

import matplotlib.pyplot as plt

path = "config.json"

def main(args):
    #---set up path for training and test data (NUAA face liveness dataset)--------------
    model_name = args.model
    with open(path) as file:
        print("Reading from json ... ")
        data = json.load(file)[model_name]
        accuracy = data['accuracy']
        model_file = data['file']
    print("Reading input from the NUAA dataset ... ")
    readd = ReadData()
    clientdir = '/content/drive/MyDrive/NormalizedFace_NUAA/ClientNormalized/'
    imposterdir = '/content/drive/MyDrive/NormalizedFace_NUAA/ImposterNormalized/'
    client_train_normaized_file = '/content/drive/MyDrive/NormalizedFace_NUAA/client_train_normalized.txt'
    imposter_train_normaized_file = '/content/drive/MyDrive/NormalizedFace_NUAA/imposter_train_normalized.txt'
    
    client_test_normaized_file = '/content/drive/MyDrive/NormalizedFace_NUAA/client_test_normalized.txt'
    imposter_test_normaized_file = '/content/drive/MyDrive/NormalizedFace_NUAA/imposter_test_normalized.txt'

    #---------------read training, test data----------------
    train_images, train_labels = readd.read_data(clientdir, imposterdir, client_train_normaized_file, imposter_train_normaized_file)
    test_images, test_labels = readd.read_data(clientdir, imposterdir, client_test_normaized_file, imposter_test_normaized_file)


    for i in range(0,1):

        #--pick one of the following models for face liveness detection---
        if model_name =='CNN':
            print("Selected CNN")
            cnn = CNNModel()  # simple CNN model for face liveness detection---
        else:
            print("Selected Inception")
            cnn = InceptionV4Model()  #Inception model for liveness detection

        if args.resume:
            print("Resuming from the best model")
            model = cnn.load_model(model_file)#to use pretrained model
        else:
            print("Starting from scratch by creating a new model")
            model = cnn.create_model()  # create and train a new model   
        print("Starting training ...")
        model = cnn.train_model(model, train_images,train_labels,test_images,test_labels, 50, accuracy, file_to_save, model_name)
      
        test_loss, test_acc = cnn.evaluate(model, test_images,  test_labels)
        print('iteration = ' + str(i) + ' ---------------------------------------------========')
    print("**************************************Done***************************************")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for Face Liveness test")
    parser.add_argument('--model', '-m', type=str, default='CNN', help='CNN or Inception')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    #add learning rate, epoch
    args = parser.parse_args()
    sys.exit(int(main(args) or 0))
    
