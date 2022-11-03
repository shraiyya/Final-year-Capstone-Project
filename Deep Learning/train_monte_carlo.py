import os
import numpy as np
import random
import time
import glob
from Bio import SeqIO
from pybedtools import BedTool
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adadelta
from sklearn import metrics
import h5py
from sklearn.model_selection import KFold
from predict import model_predict, load_dataset
from sklearn.metrics import roc_curve,roc_auc_score, auc, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

INPUT_LENGTH = 200
EPOCH = 200
BATCH_SIZE = 64
WORK_DIR = "/content/SilencerEnhancerPredict"

def plot_roc_curve(fpr, tpr, fold_no, auc): 
  plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
  plt.title('ROC curve')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.grid()
  plt.legend()
  #plt.show())
  plt.savefig(f"curves/plot_roc_curve_{fold_no}.png") 

def test_auc_acc(test_acc_per_fold,test_auc_per_fold,y_test_kf, fold_no):
  f= '/content/SilencerEnhancerPredict/examples/training_200seq_2class.hdf5.pred.data'
  with h5py.File(f, "r") as f:
      # List all groups
      print("Keys: %s" % f.keys())
      a_group_key = list(f.keys())[0]

      # Get the data
      data_pred = list(f[a_group_key])

  data_pred_class = []
  data_pred_binary_class = []
  for i in range(len(data_pred)):
    data_pred_class.append(abs(data_pred[i][0] - data_pred[i][1]))
    if data_pred[i][0] > data_pred[i][1]:
      data_pred_binary_class.append(0) #left
    else: 
      data_pred_binary_class.append(1) #right
    
  
  d2 = y_test_kf[:].tolist()
  data_orig_class = []
  for i in range(len(d2)):
    if(d2[i] == [1.0, 0.0]):
      data_orig_class.append(0.0)
    else:
      data_orig_class.append(1.0)
  acc = accuracy_score(data_orig_class, data_pred_binary_class)
  test_acc_per_fold.append(acc)
  fpr,tpr, thresh = metrics.roc_curve(data_orig_class, data_pred_class)
  aucc = metrics.auc(fpr, tpr)
  test_auc_per_fold.append(aucc)
  plot_roc_curve(fpr, tpr, fold_no, aucc) 



def train_val_divide(mat):
  len_m=len(mat)
  len_75 = int(((len_m*3)/4))
  #len_25 = len_m -  len_75
  mat_train=mat[:len_75]
  mat_val=mat[len_75:]
  return mat_train,mat_val

def run_model(data, model, save_dir):

    weights_file = os.path.join(save_dir, "model_weights.hdf5")
    model_file = os.path.join(save_dir, "single_model.hdf5")
    model.save(model_file)

    # Adadelta is recommended to be used with default values
    opt = Adadelta()

    # parallel_model = ModelMGPU(model, gpus=GPUS)
    parallel_model = model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = data["train_data"]
    Y_train = data["train_labels"]
    X_validation = data["val_data"]
    Y_validation = data["val_labels"]
    X_test = data["test_data"]
    Y_test = data["test_labels"]

    from keras.utils.np_utils import to_categorical
    '''
    Y_train = to_categorical(Y_train, num_classes=None)
    Y_test = to_categorical(Y_test, num_classes=None)
    Y_validation = to_categorical(Y_validation, num_classes=None)
    '''

    _callbacks = []
    checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    _callbacks.append(checkpointer)
    earlystopper = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    _callbacks.append(earlystopper)

    #kfold = KFold(n_splits=10, shuffle=True)

    inputs = np.concatenate((data["train_data"], data["val_data"]), axis=0)
    inputs = np.concatenate((inputs, data["test_data"]), axis=0)
    targets = np.concatenate((data["train_labels"], data["val_labels"]), axis=0)
    targets = np.concatenate((targets, data["test_labels"]), axis=0)


    fold_no = 1
    train_acc_per_fold=[]
    train_loss_per_fold=[]
    test_acc_per_fold=[]
    test_auc_per_fold=[]

    #for train_index, test_index in kfold.split(inputs):
    for i in range(10):
      #print("TRAIN:", train_index, "TEST:", test_index)
      print("fold number: ", fold_no)
      print(" ")
      dataset = pd.DataFrame({'inputs': list(inputs), 'targets': list(targets)}, columns=['inputs', 'targets'])
      dataset = dataset.sample(frac = 1)
      train, test = train_test_split(dataset, test_size=0.1, random_state=42, shuffle=True)
      trainn, val = train_test_split(train, test_size=0.11117, random_state=42, shuffle=True)

      print('train data len: ', len(trainn))
      print('test data len: ', len(test))
      print('val data len: ', len(val))
      
      X_train_kf, X_test_kf, X_val_kf =  np.stack(trainn['inputs'].to_numpy()) ,  np.stack(test['inputs'].to_numpy()) ,  np.stack(val['inputs'].to_numpy()) 
      print('shape X_train_kf : ', X_train_kf.shape)
      print("train data head: ", trainn.head(5))
      print("val data head: ", val.head(5))
      #print("inputs", len(inputs))
      y_train_kf, y_test_kf, y_val_kf = np.stack(trainn['targets'].to_numpy()),np.stack(test['targets'].to_numpy()), np.stack(val['targets'].to_numpy())
      #print("targets", len(targets))
      #print("x_train_kf", len(X_train_kf))
      #print("x_test_kf", len(X_test_kf))

      #X_train_kf_1, X_val_kf = train_val_divide(X_train_kf)
      #print("X_train_kf_1 length = ", len(X_train_kf_1))
      #print("X_val_kf length = ", len(X_val_kf))

      #y_train_kf_1, y_val_kf = train_val_divide(y_train_kf)
      #print("y_train_kf_1 length = ", len(y_train_kf_1))
      #print("y_val_kf length = ", len(y_val_kf))

      history = parallel_model.fit(X_train_kf,
                          y_train_kf,
                          batch_size=BATCH_SIZE * 1,
                          epochs=EPOCH,
                          validation_data=(X_val_kf, y_val_kf),
                          shuffle=True,
                          callbacks=_callbacks, verbose=1)

      Y_pred = parallel_model.predict(X_test_kf)
    

      auc1 = metrics.roc_auc_score(y_test_kf[:,0], Y_pred[:,0])
      auc2 = metrics.roc_auc_score(y_test_kf[:,1], Y_pred[:,1])

      scores = parallel_model.evaluate(X_test_kf,y_test_kf, verbose=0)
      print(" ")
      print(f'Score for fold {fold_no}: {parallel_model.metrics_names[0]} of {scores[0]}; {parallel_model.metrics_names[1]} of {scores[1]*100}%')
      train_acc_per_fold.append(scores[1] * 100)
      train_loss_per_fold.append(scores[0])

      # Increase fold number
      fold_no = fold_no + 1
      model_predict('/content/SilencerEnhancerPredict/examples/training_200seq_2class.hdf5', '/content/SilencerEnhancerPredict/examples/model_weights.hdf5', '/content/SilencerEnhancerPredict/examples/training_200seq_2class.hdf5.pred.data',X_test_kf)
      test_auc_acc(test_acc_per_fold,test_auc_per_fold,y_test_kf, fold_no)
      """
      train = pd.DataFrame(None)
      trainn = pd.DataFrame(None)
      test = pd.DataFrame(None)
      val = pd.DataFrame(None)
      dataset = pd.DataFrame(None)
      """
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(train_acc_per_fold)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Loss: {train_loss_per_fold[i]} - Accuracy: {train_acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(train_acc_per_fold)} (+- {np.std(train_acc_per_fold)})')
    print(f'> Loss: {np.mean(train_loss_per_fold)}')
    print('------------------------------------------------------------------------')

    with open(os.path.join(save_dir, "test_acc.txt"), "w") as of:
        of.write(str(test_acc_per_fold))
    
    with open(os.path.join(save_dir, "test_auc.txt"), "w") as of:
        of.write(str(test_auc_per_fold))

    with open(os.path.join(save_dir, "train_auc.txt"), "w") as of:
        of.write("enhancer AUC: %f\n" % auc2)
        of.write("silencer AUC: %f\n" % auc1)

    [fprs, tprs, thrs] = metrics.roc_curve(y_test_kf[:,0], Y_pred[:, 0])
    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thr = thrs[sort_ix[0]]

    [fprs, tprs, thrs] = metrics.roc_curve(y_test_kf[:,1], Y_pred[:, 1])
    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thre = thrs[sort_ix[0]]

    with open(os.path.join(save_dir, "fpr_threshold_scores.txt"), "w") as of:
        of.write("silencer 10 \t %f\n" % fpr10_thr)
        of.write("5 \t %f\n" % fpr5_thr)
        of.write("3 \t %f\n" % fpr3_thr)
        of.write("1 \t %f\n\n" % fpr1_thr)
        of.write("enhancer 10 \t %f\n" % fpr10_thre)
        of.write("5 \t %f\n" % fpr5_thre)
        of.write("3 \t %f\n" % fpr3_thre)
        of.write("1 \t %f\n" % fpr1_thre)

def load_dataset(Dfile):

    print("reading enhancers...")
    data = {}
    with h5py.File(Dfile, "r") as inf:
        for _key in inf:
            data[_key] = inf[_key][()]
    return data

def train_model(Dfile,results_dir):

    model_file = WORK_DIR + "/examples/model.hdf5"
    model = load_model(model_file)
   
    if not os.path.exists(Dfile):
        print("no data file"+Dfile)
        exit()
        
    data = load_dataset(Dfile)
    run_model(data, model, results_dir)

    
if __name__ == "__main__":

    data = sys.argv[1]
    results_dir = sys.argv[2]
    if not os.path.exists(results_dir):
         os.mkdir(results_dir)
    train_model(data,results_dir)
