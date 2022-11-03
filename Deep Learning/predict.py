import os
import numpy as np
import random
import time
import glob
from Bio import SeqIO
from pybedtools import BedTool
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam # - Works
from sklearn import metrics
import h5py
import sys

INPUT_LENGTH = 200
EPOCH = 200
BATCH_SIZE = 200
WORK_DIR = "/content/SilencerEnhancerPredict"

def model_predict(data_file, weights_file, result_file,X_test_kf):

  model_file = WORK_DIR + "/examples/model.hdf5"
  model = load_model(model_file)
  model.load_weights(weights_file)
  data = load_dataset(data_file)
  x = X_test_kf

  print("prediction on test samples ...")
  ypred = model.predict(x, batch_size=200, verbose=1)
	
  with h5py.File(result_file, "w") as of:
    of.create_dataset(name="ypred", data=ypred, compression="gzip")


def load_dataset(datafile):

  print("reading samples...")
  data = {}
  with h5py.File(datafile, "r") as inf:
    for _key in inf:
      data[_key] = inf[_key][()]

  return data

if __name__ == "__main__":

  
  data_file = sys.argv[1]
  model_file = sys.argv[2]
  result_file = data_file+".pred.data"
  
  model_predict(data_file, model_file, result_file)
