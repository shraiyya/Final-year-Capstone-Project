import random
import string
import numpy as np
from pathlib import Path


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
    generates a string of length of ACGT
    returns: string of equal ACGT
    """
    length = 296
    len4 = length / 4
    # print("len / 4 :" , len4)
    letters = ["A", "C", "G", "T"]
    letters_list = list(np.repeat(letters, len4))
    random.shuffle(letters_list)
    str1 = ""
    return str1.join(letters_list)

def non_motif_string():
    """
    generates a string of length (by get_num) of ACGT
    returns: string of equal ACGT
    """
    length = 300
    len4 = length / 4
    letters = ["A", "C", "G", "T"]
    letters_list = list(np.repeat(letters, len4))
    random.shuffle(letters_list)
    # print(" Random generated string with repetition:")
    str1 = ""
    return str1.join(letters_list)


def add_motif(motif):
    """
    adds a motif - string input, to output by non_motif_string
    returns: string consisting of a motif
    """
    # motif = 'ATCAAG'
    result = list(motif_string())
    # print(''.join(result))
    i = random.choice(range(len(result)))
    # print(i)
    result.insert(i, motif)
    result = "".join(result)
    return result


def training_write_file(length): #for motif and non motif
    """
    generate dataset
    input:
            length: number of rows
            add_motif_option = True/False
    returns:  a .txt file
    """
    l = []
    name = ""
    motifno = ""
    for i in range(int(length)):

        if i%2==0:
            name = "motif"
            result = add_motif("ATCAAG")
            motifno = "1"
        else:
            name = "non_motif"
            result = non_motif_string()
            motifno = "0"
        seq = "seq_" + str(i + 1) + "_peak"
        l.append("A" + '\t' + seq + '\t' + result + '\t' + motifno)
        #l.append(seq)
        #l.append(result)
        #l.append(motifno)

    filepath = Path("Data/datasetpy/"+ name + ".txt")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding ="utf-8") as f:
        for s in l:
            f.write( str(s) + "\n")

def testing_write_file(length):
    """
    generate dataset
    input:
            length: number of rows
    returns:  a .txt file
    """
    l = []
    name = ""
    motifno = ""
    name="seq_testing1000"
    for i in range(int(length)):
            result = add_motif("ATCAAG")
            motifno = "1"
            seq = "seq_" + str(i + 1) + "_peak"
            l.append("A" + '\t' + seq + '\t' + result + '\t' + motifno)
    for i in range(int(length)):
            result = non_motif_string()
            motifno = "0"
            seq = "seq_" + str(i + 1) + "_shuf"        
            l.append("A" + '\t' + seq + '\t' + result + '\t' + motifno)
        #l.append(seq)
        #l.append(result)
        #l.append(motifno)

    filepath = Path("Data/datasetpy/"+ name + ".txt")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding ="utf-8") as f:
        for s in l:
            f.write( str(s) + "\n")


def training_write_file_only_motif(length): #for motif only
    """
    generate dataset
    input:
            length: number of rows
            add_motif_option = True/False
    returns:  a .txt file
    """
    l = []
    name = ""
    motifno = ""
    i=-1
    result=""
    for i1 in range(int(length)):
        name = "motif"
        result,i = add_motif("ATACGTTACCCG")
        seq = "seq_" + str(i1 + 1) + "_peak"
        l.append("A" + '\t' + seq + '\t' + result+ '\t' + '1')
        #l.append(seq)
        #l.append(result)
        #l.append(motifno)

    filepath = Path("Data/datasetpy/"+ name + ".txt")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding ="utf-8") as f:
        for s in l:
            f.write( str(s) + "\n")

def training_write_file_only_non_motif(length): #for non_motif only
    """
    generate dataset
    input:
            length: number of rows
            add_motif_option = True/False
    returns:  a .txt file
    """
    l = []
    name = ""
    motifno = ""
    i=-1
    result=""
    for i1 in range(int(length)):
        name = "non-motif"
        result = non_motif_string()
        seq = "seq_" + str(i1 + 1) + "_peak"
        l.append("A" + '\t' + seq + '\t' + result+ '\t' + '0')
        #l.append(seq)
        #l.append(result)
        #l.append(motifno)

    filepath = Path("Data/datasetpy/"+ name + ".txt")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding ="utf-8") as f:
        for s in l:
            f.write( str(s) + "\n")


#training_write_file_only_motif(5084)
#training_write_file(5084)
#testing_write_file(500)