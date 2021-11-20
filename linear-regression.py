import numpy as np
from matplotlib import pyplot as plt

from myIO import read_dataset_with_pandas
from preprocessing import range_min_to_max, zero_mean_unit_variance

# colName, data = read_dataset_with_pandas('dataset/Data-Train.csv')
colName, data = read_dataset_with_pandas('dataset/Data-Train-mini.csv')
data = data.to_numpy()
X = np.array(list(map(lambda x:x[:-1],data)))
y = np.array(list(map(lambda x:x[-1],data)))

