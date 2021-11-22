from myIO import read_dataset_with_pandas
from normalization import clipping, log_scaling, range_min_to_max, zero_mean_unit_variance

def read_dataset_to_X_and_y(file, atr= None, normalization = None, minValue = None, maxValue = None):
    """
    Read the attribute(atr) that you want and put X0 = 1 and thoes attribute 
    of all samples in X and all samples lable in y 
    normalization:
    .   by default is None and can be "z_score", "scaling", "clipping"
        or "log_scaling"
    .   for "scaling", "clipping" must set minValue and
        maxNormalization
    Return X and y as nparray
    """
    import numpy as np
    colName, data = read_dataset_with_pandas(file)
    data = data.to_numpy()

    if(normalization != None):
        if(normalization == 'z_score'):
            data = zero_mean_unit_variance(data)
        elif(normalization == 'scaling'):
            data = range_min_to_max(data, minValue, maxValue)
        elif(normalization == 'clipping'):
            data = clipping(data, minValue, maxValue)
        elif(normalization == 'log_scaling'):
            data = log_scaling(data)
        else:
            print('method should be "z_score", "scaling", "clipping" or "log_scaling"')
            return 

    if(atr == None):
        samples = np.array(list(map(lambda x:np.concatenate(([1], x[:-1])),data)))
        lable = np.array(list(map(lambda x:[x[-1]],data)))
    else:
        if(atr[1] == len(colName)):
            atr[1] -= 1
        samples = np.array(list(map(lambda x:np.concatenate(([1], x)),data)))
        lable = np.array(list(map(lambda x:[x[-1]],data)))
    return samples, lable
