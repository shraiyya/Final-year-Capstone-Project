from re import A
from Bio import SeqIO
import pandas as pd

def fasta_to_dataframe(bound,name, file1):
  with open(file1) as fasta_file:  # Will close handle cleanly
    identifiers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        identifiers.append(seq_record.id)
        seqs.append(str(seq_record.seq))

    data= {'EventId':identifiers,'seq':seqs}
    df=pd.DataFrame(data)
    df['FoldID'] ='A'
    df['start_index']='-5'
    df=df[["FoldID","EventId","start_index","seq"]] #ordering the columns
    df['seq']=df['seq'].str.upper()
    df['Bound'] =bound
    df.to_csv(name + '.txt', index=None, sep='\t')
    return df

#df3=fasta_to_dataframe(1,"tad", '/content/tad.fasta')
#df3=fasta_to_dataframe(0,"left", '/content/left.fasta')
#df3=fasta_to_dataframe(0,"right", '/content/right.fasta')
