import pandas as pd 
import numpy as np

colnames=['FoldID', 'EventID', 'seq', 'Bound'] 
df = pd.read_csv('Data/datasetpy/motif.txt', delimiter = ' ',  names = colnames, header=None)
#print(df.head())

df.set_index('FoldID')
#print(np.where(pd.isnull(df)))
#print(df.count())
df.to_csv(r'Data/datasetpy/sampledataset2onlymotif.seq', index=None, sep='\t', mode='a')

print(df.dtypes)
