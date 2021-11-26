from my_io import read_dataset_with_pandas
from normalization import clipping, log_scaling, range_min_to_max, zero_mean_unit_variance

def read_dataset_to_X_and_y(file, atr= None, normalization = None, min_value = None, max_value = None):
    """
    Read the attribute(atr) that you want and put X0 = 1 and thoes attribute 
    of all samples in X and all samples lable in y 
    normalization:
    .   by default is None and can be "z_score", "scaling", "clipping"
        or "log_scaling"
    .   for "scaling", "clipping" must set min_value and max_value
    Return X and y as nparray
    """
    import numpy as np
    col_name, data = read_dataset_with_pandas(file)
    data = data.to_numpy()

    if(atr == None):
        samples = np.array(list(map(lambda x:np.concatenate(([1], x[:-1])),data)))
        lable = np.array(list(map(lambda x:[x[-1]],data)))
    else:
        if(atr[1] == len(col_name)):
            atr[1] -= 1
        samples = np.array(list(map(lambda x:np.concatenate(([1], x)),data)))
        lable = np.array(list(map(lambda x:[x[-1]],data)))

    if(normalization != None):
        if(normalization == 'z_score'):
            samples = zero_mean_unit_variance(samples)
        elif(normalization == 'scaling'):
            samples = range_min_to_max(samples, min_value, max_value)
        elif(normalization == 'clipping'):
            samples = clipping(samples, min_value, max_value)
        elif(normalization == 'logScaling'):
            samples = log_scaling(samples)
        else:
            print('method should be "z_score", "scaling", "clipping" or "logScaling"')
            return 
    samples[:,0] = 1
    return samples, lable
