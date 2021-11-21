import numpy as np
from matplotlib import pyplot as plt

from myIO import read_dataset_to_X_and_y

def closed_form(XInput,yInput):
    return np.linalg.inv(XInput.transpose()@XInput)@XInput.transpose()@yInput

def linear_regression(fileTrain, fileTest, method, atr = None, normalization = None, minNormalization = None, maxNormalization = None, ploter=0, printer = 0):
    """
    It gets two dataset of train and test data and method.
    method:
    .   can be "closed_form" or "gradient_descent"
    normalization:
    .   by default is None and can be "Z-score", "scaling", "clipping"
        or log_scaling
    .   for "scaling", "clipping" must set minNormalization and
        maxNormalization
    Return learned parameters (θ0 , θ1 , ..., θn ) and the value of MSE 
    error on the train and test data.
    """
    XTrain,yTrain = read_dataset_to_X_and_y(fileTrain)
    if(printer):
        print(f'matrix XTrain is\n{XTrain}\n')
    if(printer):
        print(f'vector yTrain is\n{yTrain}\n')

    if(method == 'closed_form'):
        teta = closed_form(XTrain,yTrain)
    elif(method == 'gradient_descent'):
        teta = closed_form(XTrain,yTrain)
    else:
        print('method should be "closed_form" or "gradient_descent"')
        return    
    if(printer):
        print(f'vector learned parameters (θ0 , θ1 , ..., θn ) is\n{teta}\n')

    predictionTrain = XTrain @ teta
    if(printer):
        print(f'vector predictionTrain is\n{predictionTrain}\n')

    errorTrain = predictionTrain - yTrain
    if(printer):
        print(f'vector errorTrain is\n{errorTrain}\n')

    MSETrain = (np.square(errorTrain)).mean(axis=0)
    if(printer):
        print(f'MSE on train data is\n{MSETrain}\n')

    XTest,yTest = read_dataset_to_X_and_y(fileTest)
    if(printer):
        print(f'matrix XTest is\n{XTest}\n')
    if(printer):
        print(f'vector yTest is\n{yTest}\n')

    # exit()
    predictionTest = XTest @ teta
    if(printer):
        print(f'vector predictionTest is\n{predictionTest}\n')

    errorTest = predictionTest - yTest
    if(printer):
        print(f'vector errorTest is\n{errorTest}\n')

    MSETest = np.square(errorTest).mean(axis=0)
    if(printer):
        print(f'MSE on test data is\n{MSETest}\n')


    if(ploter):
        sXTest = XTest.argmin(axis=0)[1]
        eXTest = XTest.argmax(axis=0)[1]
        plt.plot(list(zip(*XTest))[1],yTest,'.',lable = 'datasets test')
        plt.plot([XTest[sXTest][1],XTest[eXTest][1]],[predictionTest[sXTest],predictionTest[eXTest]],'-r',lable = 'regression line')
        plt.xlabel('Feature')
        plt.ylabel('Lable')
        plt.show()
    
    return teta, MSETrain, MSETest

teta, MSETrain, MSETest = linear_regression('dataset/Data-Train.csv','dataset/Data-Test.csv', method='closed_form')
print(f'vector learned parameters (θ0 , θ1 , ..., θn ) is\n{teta}\n')
print(f'MSE on train data is\n{MSETrain}\n')
print(f'MSE on test data is\n{MSETest}\n')

