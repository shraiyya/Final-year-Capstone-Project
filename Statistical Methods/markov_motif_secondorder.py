import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import accuracy_score


def combinations_list(max_length):

    """
    returns list of combinations of ACGT of all lengths possible 
    """
    letters = ["0", "A", "C", "G", "T"]
    # max_length = 4
    b = len(letters) - 1
    #  base to convert to
    n = 0
    k = 0
    while k < max_length:
        n = (n * b) + b
        k += 1
    #  number of combinations
    i = 1
    l = []
    while i <= n:
        current = i
        #  m and q_n in the formula
        combination = ""
        while True:
            remainder = current % b
            if remainder == 0:
                combination += letters[b]
                current = int(current / b) - 1
            else:
                combination += letters[remainder]
                current = int(current / b)
            if (current > 0) == False:
                break
        l.append(combination)
        i += 1
    return l


def markov_model_motif(df1):
    """
    for a second degree motif markov model 
    returns dict2, dict3: count of the appearance of eg: A->C and eg: AC->T occuring
    """
    dict2 = {}
    dict3 = {}
    max_length = 3
    l = combinations_list(max_length)
    for ele in l:
        if len(ele) == 2:
            dict2[ele] = 0
        elif len(ele) == 3:
            dict3[ele] = 0

    for j in range(len(df1)):
        seq = df1[j]
        # print(seq)
        # print(" ")
        # print(seq)

        for k in range(0, len(seq) - 2):
            if seq[k : k + 2] in dict2.keys():
                dict2[seq[k : k + 2]] += 1
        # print(dict2)

        for i in range(0, len(seq)):
            if seq[i : i + 3] in dict3.keys():
                dict3[seq[i : i + 3]] += 1
        # print(dict3)
        # print(" ")

    for i1 in dict3:
        if i1[0:2] in dict2:
            if dict2[i1[0:2]] != 0:
                dict3[i1] = dict3[i1] / dict2[i1[0:2]]

    return dict2, dict3


def motif_score(dict2, dict3, seq):
    """
    returns the score of th markov model motif 
    """
    s = seq
    import math

    score = 1
    for i in range(0, len(s)):
        if s[i : i + 3] in dict3:
            score = score * dict3[s[i : i + 3]]
    return math.log(score)


def train():
    # df1 = df['seq']
    seq_list = df1.to_list()  # list of sequences
    dict2, dict3 = markov_model_motif(df1)  # markov model for the motif.txt file
    # print("####################################################")
    print(dict2)
    print(dict3)
    dict2_non, dict3_non = markov_model_motif(
        df3
    )  # markov model for the non motif file
    # print("####################################################")
    # print(dict2_non)
    # print(dict3_non)

    # now, testing each sequence and predicting their class using the 2 dictionaries

    for i in range(len(df4)):
        list1 = []
    for i in range(len(df4)):
        seq = df4["seq"][i]
        """
        1. score it using ddict3
        2. score it using dict3_non 
        3. whichever score higher, give value of that class 
        4. add seq, class to a list, append that list to the df 
        5. write this file 
        """

        score_motif = motif_score(dict2, dict3, seq)
        print(seq)
        print(score_motif)
        score_nonmotif = motif_score(dict2_non, dict3_non, seq)
        print(score_nonmotif)

        if score_motif > score_nonmotif:
            list1.append(1)
        else:
            list1.append(0)
    df4["predicted"] = list1
    print(df4)


def accuracy():
    y_true = df4["Bound"]
    y_pred = df4["predicted"]
    return accuracy_score(y_true, y_pred)


colnames = ["FoldID", "EventID", "start_index", "seq", "Bound"]
df = pd.read_csv(
    "Data/markov_files/motif.txt", delimiter="\t", names=colnames, header=None
)
colnames2 = ["FoldID", "EventID", "seq", "Bound"]
df2 = pd.read_csv(
    "Data/markov_files/non-motif.txt", delimiter="\t", names=colnames2, header=None
)
df1 = df["seq"]
df3 = df2["seq"]
df1.reset_index(drop=True, inplace=True)
df3.reset_index(drop=True, inplace=True)
df4 = pd.read_csv(
    "Data/markov_files/mixed.txt", delimiter="\t", names=colnames, header=None
)
train()
acc = accuracy()
print(acc)
# dict2, dict3 = markov_model_motif(df1)
# score = motif_score(dict2, dict3)
# print(dict2)
# print(dict3)
# print(score)
