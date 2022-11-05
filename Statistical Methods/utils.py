import pandas as pd
import tensorflow as tf

df=pd.read_csv('/content/deepbind_scores_1.csv')
df

softmax = tf.nn.softmax(df)
df1 = pd.DataFrame(np.array(softmax))
df1

df1.to_csv("deepbind_scores_softmax_output.csv")
