from myIO import readDatasetWithPandas
from normalization import clipping, logScaling, rangeMinToMax, zScore

def readDatasetToXAndY(file, atr= None, normalization = None, minValue = None, maxValue = None):
    """
    Read the attribute(atr) that you want and put X0 = 1 and thoes attribute 
    of all samples in X and all samples lable in y 
    normalization:
    .   by default is None and can be "z_score", "scaling", "clipping"
        or "logScaling"
    .   for "scaling", "clipping" must set minValue and
        maxNormalization
    Return X and y as nparray
    """
    import numpy as np
    colName, data = readDatasetWithPandas(file)
    data = data.to_numpy()

    if(atr == None):
        samples = np.array(list(map(lambda x:np.concatenate(([1], x[:-1])),data)))
        lable = np.array(list(map(lambda x:[x[-1]],data)))
    else:
        if(atr[1] == len(colName)):
            atr[1] -= 1
        samples = np.array(list(map(lambda x:np.concatenate(([1], x)),data)))
        lable = np.array(list(map(lambda x:[x[-1]],data)))

    if(normalization != None):
        if(normalization == 'z_score'):
            samples = zScore(samples)
        elif(normalization == 'scaling'):
            samples = rangeMinToMax(samples, minValue, maxValue)
        elif(normalization == 'clipping'):
            samples = clipping(samples, minValue, maxValue)
        elif(normalization == 'logScaling'):
            samples = logScaling(samples)
        else:
            print('method should be "z_score", "scaling", "clipping" or "logScaling"')
            return 

    return samples, lable
