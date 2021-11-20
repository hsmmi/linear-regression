import numpy as np
from matplotlib import pyplot as plt

from myIO import read_dataset_with_pandas

# colName, data = read_dataset_with_pandas('dataset/Data-Train.csv')
colName, data = read_dataset_with_pandas('dataset/Data-Train-mini.csv')
data = data.to_numpy()
X = np.array(list(map(lambda x:x[:-1],data)))
y = np.array(list(map(lambda x:x[-1],data)))

