import random
import random
import string
import numpy as np
from pathlib import Path
import pandas as pd
import copy
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import pylab as pl
from sklearn import metrics
from sklearn.model_selection import KFold
import os 
import seaborn as sns
random.seed(10)



def get_num(a, b, x):
    """
    gets a unique number between a range (a,b), divisible by x
    input:  a - start inde
            b - end index
            x = random number should be divisible by x
    return:
        random number divisible by x
    """
    if not a % x:
        return random.choice(range(a, b, x))
    else:
        return random.choice(range(a + x - (a % x), b, x))

def motif_string():
    """
    generates a string of length (by get_num) of ACGT
    returns: string of equal ACGT
    """
    length = 288
    # print("random length: " , length)
    len4 = length / 4
    # print("len / 4 :" , len4)
    letters = ["A", "C", "G", "T"]
    letters_list = list(random.choice(letters))
    random.shuffle(letters_list)
    # print(" Random generated string with repetition:")
    str1 = ""
    return str1.join(letters_list)

def non_motif_string(length):
    """
    generates a string of length (by get_num) of ACGT
    returns: string 
    """
    # print("random length: " , length)
    #len4 = length / 4
    # print("len / 4 :" , len4)
    letters = ["A", "C", "G", "T"]
    nonmotiflist = []
    for i in range(length): 
      nonmotiflist.append(random.choice(letters))
    # print(" Random generated string with repetition:")
    str1 = ""
    return str1.join(nonmotiflist)

def add_motif():
    """
    adds a motif - string input, to output by non_motif_string
    returns: string consisting of a motif
    """
    motif = 'ATACGTTACCCG'
    result = list(motif_string())
    i = random.choice(range(len(result)))
    result.insert(i, motif)
    result = "".join(result)
    return result,i


def write_to_df(length,var): #to generate non motif background files
  l=[]
  for i1 in range(int(length)):
    l1=[]
    i=-5
    result= non_motif_string(var)
    seq = "seq_" + str(i1 + 1) + "_peak"
    a='0'      
    l1 = ["A",seq,str(i), result,a]
    l.append(list(l1))
  df = pd.DataFrame(l,columns=['FoldID',	'EventID',	'start_index',	'seq',	'Bound']) 
  return df


def read_pfm_jaspar(filepath_):
  with open(Path(filepath_)) as f:
    lines = f.readlines()

  a = []
  for i in range(1,5,1):
    b = list(lines[i].split()[2:-1])
    a.append(b)
  a = np.array(a,dtype=float)
  return a

def convert_pfm_to_ppm(pfm):
  for i in range(pfm.shape[1]):
    sum = 0
    for j in range(pfm.shape[0]):
      sum+= pfm[j][i] 
    for j in range(pfm.shape[0]):
      prob = float(pfm[j][i]/sum)
      ppm[j][i] = float(prob)
  return(ppm)
  
p1=['A','C','G','T']
str1=''
def ppm_to_motif(ppm):
  motif = []
  mot=[]
  for i in range(ppm.shape[1]):
    arr = []
    for j in range(ppm.shape[0]):
      arr.append(float(ppm[j][i]))
    choice = np.random.choice(p1,p=arr) #motif is generated here
    mot.append(choice)
  
  return (''.join(mot))

def non_motif_create(df1,df2): #to implant a  motif in a non motif file -> 1. motif file, 2. mixed file 
  mot=[]
  l1=[]
  l2=[]
  for i in range(0,len(df1)):
    motif = ppm_to_motif(ppm)
    mot.append(motif)
    index=random.randint(0,len(df1[i])-1) #0 to 300 
    l1.append(index)
    df1[i]=df1[i][:index] + str(motif) + df1[i][index:]
    
    index2=np.random.choice(len(df2[i]))
    l2.append(index2)
    df2[i]=df2[i][:index2] + str(motif) + df2[i][index2:]
    #print(index)
    #print(df1[i])
    #print(mot)
  return l1,l2,df1,df2,mot #l1, l2 to store the index

def markov_model_motif(df1):
    """
    for a second degree motif markov model 
    returns dict2, dict3: count of the appearance of eg: A->C and eg: AC->T occuring
    """
    
    for j in range(len(df1)):
      seq = df1[j]
      for i in range(len(seq)-var): #dict2
        if seq[i:i+var] not in dict2: 
          dict2[seq[i:i+var]] = 1
        else: 
          dict2[seq[i:i+var]] += 1

      for i in range(len(seq)): #dict3
        if seq[i:i+var+1] not in dict3: 
          dict3[seq[i:i+var+1]] = 1
        else: 
          dict3[seq[i:i+var+1]] += 1

    #print('dict2: ', dict2)
    #print('dict3: ', dict3)
      
    return dict2, dict3

def motif_score1(dict2, dict3, seq):
    """
    returns the score of th markov model motif 
    """
    s = seq
    import math

    a=0
    sum = 0
    score = 1
    for i in range(0, len(s)):
      if s[i : i + var+1] not in dict3:
        #print("HELLOOOOOOOOOOOO", s[i : i + var+1])
        dict3[s[i : i + var+1]] = 1
      score = score * dict3[s[i : i + var+1]]
    #print('dict2-',dict2)
    #print('dict3-',dict3)
    return math.log(score),dict2,dict3

def motif_score(dict2, dict3, seq):
    """
    returns the score of th markov model motif 
    """
    s = seq
    import math
    score = 0
    ep=0.01
    for i in range(0, len(s)):
      if s[i : i + var+1] not in dict3:
        #print("HELLOOOOOOOOOOOO", s[i : i + var+1])
        dict3[s[i : i + var+1]] = ep
      else:
        dict3[s[i : i + var+1]] = dict3[s[i : i + var+1]] + ep
      score = score + math.log(dict3[s[i : i + var+1]])

    #print('dict2-',dict2)
    #print('dict3-',dict3)
    return score,dict2,dict3

def train(df,df2,df4):
  """
  df = for the motif markov model - seq 
  df2 = for the non motif markov model  - seq 

  df4 = the test file - pass the entire thing = seq + bound 

  Returns: the final dictionaries
  """
    # df1 = df['seq']
   
  dict2, dict3 = markov_model_motif(df)  # markov model for the motif.txt file
  # print("####################################################")
  
  dict2_non, dict3_non = markov_model_motif(df2)  # markov model for the non motif file
  # now, testing each sequence and predicting their class using the 2 dictionaries

  for i in range(len(df4)):
      list1 = []
      list2 = []
  for i in range(len(df4)):
      seq = df4["seq"][i]
      """
      1. score it using ddict3
      2. score it using dict3_non 
      3. whichever score higher, give value of that class 
      4. add seq, class to a list, append that list to the df 
      5. write this file 
      """

      score_motif,dict5,dict6 = motif_score(dict2, dict3, seq)
      score_nonmotif,dict7,dict8 = motif_score(dict2_non, dict3_non, seq)
      #print('for sequence: ', seq, 'score motif: ', score_motif, 'score_nonmotif: ', score_nonmotif)
      list1.append(score_motif-score_nonmotif)
      if score_motif > score_nonmotif:
          list2.append(1)
      else:
          list2.append(0)
  df4["predicted"] = list2
  df4["log odd score"] = list1
  print(df4)
  if score_motif > score_nonmotif:
    return dict6
  else:
    return dict8

  #print('dict2 - motif: ', dict5)
  #print('dict3 - motif : ', dict6)
 
def accuracy(df4):
    y_true = df4["Bound"]
    y_pred = df4["predicted"]
    return accuracy_score(y_true, y_pred)

def acc_score(df4):
  y_true=df4['Bound']
  y_pred=df4['predicted']
  accuracy_score(y_true, y_pred)
  return y_true, y_pred



def confusion_mat(y_true,y_pred):
  array3=confusion_matrix(y_true, y_pred)

  plt.figure(figsize=(5,5))   
  sns.heatmap(array3, annot=True,fmt='.2f',cmap="PuBu")
  plt.title('Confusion Matrix',fontsize=15,color='red')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()


def auc_calculate(df4):
  y_log_score = df4['log odd score']
  precision, recall, thresholds = precision_recall_curve(y_true, y_log_score)
  area = auc(recall, precision)
  print("Area Under Curve")
  print(area)

  pl.clf()
  pl.plot(recall, precision, label='Precision-Recall curve')
  pl.xlabel('Recall')
  pl.ylabel('Precision')
  pl.ylim([0.0, 1.0])
  pl.xlim([0.0, 1.0])
  pl.title('Precision-Recall example: AUC=%0.2f' % area)
  pl.legend(loc="lower left")
  pl.show()

def roc_calculate(df4):
  y_log_score = df4['log odd score']
  fpr,tpr, thresh = metrics.roc_curve(y_true, y_log_score)
  auc = metrics.auc(fpr, tpr)
  print("AUC:", auc)

  plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
  plt.title('ROC curve')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.grid()
  plt.legend()
  plt.show()









df_nonmotif_train = write_to_df(1000,314)
df_mixed_motif = write_to_df(1000,300)
df_mixed_nonmotif = write_to_df(1000,314)
df_motif_train = write_to_df(1000,300)
print('###### PRINTING NON MOTIF TRAIN DATAFRAME ########')
print(df_nonmotif_train.head())
pfm = read_pfm_jaspar('MA0003.4.jaspar')
print("##### PRINTING PFM #####")
print(pfm)
ppm = copy.deepcopy(pfm)
print("##### PRINTING PPM DEEPCOPY #####")
print(ppm)
ppm = convert_pfm_to_ppm(pfm)
print("##### PRINTING PPM  #####")
print(ppm)
motif= ppm_to_motif(ppm)
print("##### PRINTING ONE MOTIF  #####")
print(motif)

l1,l2,df_motif_train['seq'],df_mixed_motif['seq'],mot=non_motif_create(list(df_motif_train['seq']),list(df_mixed_motif['seq']))
print('##### l1 length #####')
print(l1)
print('##### len(df_motif_train[seq][0]) #####')
print(len(df_motif_train['seq'][0]))

df_motif_train['start_index'] = l1
df_mixed_motif['start_index'] = l2
df_motif_train['Bound']=df_motif_train['Bound'].astype('int32').replace(0,1)
df_mixed_motif['Bound']=df_mixed_motif['Bound'].astype('int32').replace(0,1)

#df_mixed_motif['Bound'].dtype
print('##### df_mixed_motif head #####')
print(df_mixed_motif.head())
df_mixed_motif=df_mixed_motif.append(df_mixed_nonmotif,ignore_index=True)
df_mixed=df_mixed_motif.sample(frac=1).reset_index(drop=True)
print('##### df mixed #####')
print(df_mixed.head())

path = Path('df_to_csv/MA0003.4')
if path.exists() == False:
    print('hello')
    os.mkdir(path)

df_mixed.to_csv(str(path) +'/mixed.txt',index=None, sep='\t')
df_motif_train.to_csv(str(path) +'/motif.txt',index=None, sep='\t')
df_nonmotif_train.to_csv(str(path) +'/nonmotif.txt',index=None, sep='\t')




dict2, dict3 = {}, {}
dict2non, dict3non = {}, {}

colnames2 = ["FoldID", "EventID", "start_index", "seq", "Bound"]
colnames = ["FoldID", "EventID", "seq", "Bound"]

#df =  motif 
#df2 = non motif
#df4 = mixed test file
df = pd.read_csv(str(path) +'/motif.txt', delimiter="\t")

df2 = pd.read_csv(str(path) +'/nonmotif.txt', delimiter="\t")
df1 = list(df["seq"])
df3 = list(df2["seq"])
#df1.reset_index(drop=True, inplace=True)
#df3.reset_index(drop=True, inplace=True)
df4 = pd.read_csv(str(path) +'/mixed.txt', delimiter="\t")

#df4=df4.rename(columns=df4.iloc[0]).drop(df4.index[0])
#df4 = df4.sample(frac=1).reset_index(drop=True)
print('##### df head #####')
print(df4.head())

motif_seq = df['seq'].to_list()
nonmotif_seq = df2['seq'].to_list()
print("Len of motif list: ", len(motif_seq))
print("Len of non motif list: ", len(nonmotif_seq))
X_motifs = motif_seq
y_motifs = [1 for i in range(len(motif_seq))]

X_nonmotifs = nonmotif_seq
y_nonmotifs = [1 for i in range(len(nonmotif_seq))]

kf = KFold(n_splits=10)
print(kf.get_n_splits(X_motifs))

all_train_index_motif, all_test_index_motif  = [], []
all_train_index_nonmotif, all_test_index_nonmotif  = [], []

#motif
for train_index, test_index in kf.split(X_motifs): 
  #print("TRAIN: ", train_index, "TEST: ", test_index)
  all_train_index_motif.append(train_index)
  all_test_index_motif.append(test_index)

#nonmotif
for train_index, test_index in kf.split(X_nonmotifs): 
  #print("TRAIN: ", train_index, "TEST: ", test_index)
  all_train_index_nonmotif.append(train_index)
  all_test_index_nonmotif.append(test_index)

print('##### len(all_train_index_motif[0]) #####')
print(len(all_train_index_motif[0]))
print('##### len(all_train_index_motif) #####')
print(len(all_train_index_motif))
print('##### len(all_train_index_nonmotif) #####')
print(len(all_train_index_nonmotif))
print('##### all_train_index_nonmotif[0] #####')
len(all_train_index_nonmotif[0])
print('##### all_test_index_nonmotif #####')
len(all_test_index_nonmotif)
print('##### all_test_index_motif[0]) #####')
len(all_test_index_motif[0])

for i in range(10):
  l_motif=[]
  l_nonmotif=[]

  motif_seq=[]
  for a in all_train_index_motif[i]:
    motif_seq.append(X_motifs[a])

  nonmotif_seq=[]
  for a1 in all_train_index_nonmotif[i]:
    nonmotif_seq.append(X_nonmotifs[a1])
  

  for j in range(len(all_test_index_motif[0])):
    test_set_motif=X_motifs
    l_motif.append(X_motifs[all_test_index_motif[i][j]])

    test_set_nonmotif=X_nonmotifs
    l_nonmotif.append(X_nonmotifs[all_test_index_nonmotif[i][j]])

  l_1 =[1 for i in range(len(l_motif))]
  l_0 =[0 for i in range(len(l_nonmotif))]

  df_test= pd.DataFrame(list(zip(l_motif,l_1)),columns=['seq','Bound'])
  extra={'seq':l_nonmotif,'Bound':l_0}
  df_test= df_test.append(pd.DataFrame(extra))
  df_test=df_test.sample(frac=1).reset_index(drop=True)
  #print(df_test)

  for i1 in range(2,5,1):
    print("Step: ", i1)
    var = i1
    dict6 = train(motif_seq,nonmotif_seq,df_test)
    print("dict6",dict6)
    print("TCT: ",dict6['TCT'])
    print("CTC: ",dict6['CTC'])
    print("TCG ",dict6['TCG'])
    acc = accuracy(df_test)
    print(acc)  
    y_true,y_pred = acc_score(df_test)
    confusion_mat(y_true,y_pred)
    #auc_calculate(df_test)
    roc_calculate(df_test)





