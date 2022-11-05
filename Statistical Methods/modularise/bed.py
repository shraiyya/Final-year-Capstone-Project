import pandas as pd 
import numpy as np

#flag=1 -> tad file 
#flag=2 -> left boundry file 
#flag=3 -> rightboundry file 


def bed_file(df, flag,name,shift):
  if flag==1:
    #tad
    tad = (df['chr1_num'] + df['chr2_num']) / 2
    df2 = tad.to_frame()
    df2 = df2.astype('int')
    left = df2[0] - shift 
    left.to_frame()
    right = df2[0] + shift 
    right.to_frame()
    chrom_num = df['chromosome_num']
    df = df.join(left).rename(columns={0: 'left'})
    df = df.join(right).rename(columns={0: 'right'})
    df['.'] = df['chromosome_num'] + str(':') + df['left'].astype('str') + str('-') + df['right'].astype(str)

  elif flag==2:
    #left
    lbx = (df['chr1_num'] - shift-200 )
    lbx=lbx.to_frame()
    lbx=lbx.rename(columns={'chr1_num':'LeftBoundaryX'})
    df=df.join(lbx)
    lby = (df['chr1_num'] - shift )
    lby=lby.to_frame()
    lby=lby.rename(columns={'chr1_num':'LeftBoundaryY'})
    df=df.join(lby)
    df['.'] = df['chromosome_num'] + str(':') +df['LeftBoundaryX'].astype('str') + str('-') + df['LeftBoundaryY'].astype(str)

  elif flag==3:
    #right
    rbx = (df['chr2_num'] + shift )
    rbx=rbx.to_frame()
    rbx=rbx.rename(columns={'chr2_num':'RightBoundaryX'})
    df=df.join(rbx)
    rby = (df['chr2_num'] + shift + 200 )
    rby=rby.to_frame()
    rby=rby.rename(columns={'chr2_num':'RightBoundaryY'})
    df=df.join(rby)
    df['.'] = df['chromosome_num'] + str(':') +df['RightBoundaryX'].astype('str') + str('-') + df['RightBoundaryY'].astype(str)

  else: 
    print("error")
    return
  df = df.drop(columns=['chr1_num','chr2_num'])
  df.to_csv(name + '.bed', header = None, index=None, sep='\t')




arr = ['chromosome_num', 'chr1_num', 'chr2_num']
df = pd.read_csv('/content/GSE101317_S2Rplus_G1S_domain.txt', sep = '\t', names= arr)

#flag=1 -> tad file 
#flag=2 -> left boundry file 
#flag=3 -> rightboundry file 

#shift is the original shit. 
#eg for left boundry shift = 500, then we'll be finding coordinares from -700 to -500
#bed_file(df,flag=1,name="tad",shift=100)
#bed_file(df,flag=2,name="left",shift=800)
#bed_file(df,flag=3,name="right",shift=800)

