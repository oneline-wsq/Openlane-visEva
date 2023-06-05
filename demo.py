from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np

a=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

a= np.array(a)

sort_idx=np.argsort(a)
print(a)
print(sort_idx)