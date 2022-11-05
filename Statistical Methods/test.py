from pathlib import Path 
import pandas as pd 
import os 

path = Path('df_to_csv/MA0003.4')
#obj = Path(path)

if path.exists() == False:
    print('hello')
    os.mkdir(path)


lst = [['tom', 25], ['krish', 30],
       ['nick', 26], ['juli', 22]]
    
df = pd.DataFrame(lst, columns =['Name', 'Age'])

#df.to_csv(obj)
df.to_csv(str(path) + '/mixed.txt',index=None, sep='\t')