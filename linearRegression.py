import numpy as np
from matplotlib import pyplot as plt

from preprocessing import read_dataset_to_X_and_y

def closed_form(XInput,yInput):
    return np.linalg.inv(XInput.transpose()@XInput)@XInput.transpose()@yInput

def linear_regression(XTrain, yTrain, method, printer = 0):
    """
    It gets matrix samples(XTrain) and their labels(yTrain) and method.
    method:
    .   can be "closed_form" or "gradient_descent"
    Return learned parameters (θ0 , θ1 , ..., θn ) and the value of MSE 
    error on the train and test data.
    """
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
    
    return teta, MSETrain

def linear_regression_evaluation(XTest, yTest, teta, printer = 0, ploter=0):
    if(printer):
        print(f'matrix XTest is\n{XTest}\n')
    if(printer):
        print(f'vector yTest is\n{yTest}\n')

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

    return MSETest

normalizationMethod = [None ,"z_score", "scaling", "clipping", "log_scaling"]
i = 1

XTrain, tTrain = read_dataset_to_X_and_y('dataset/Data-Train-mini.csv',normalization=normalizationMethod[i])

teta, MSETrain = linear_regression(XTrain, tTrain , method='closed_form', printer=1)

XTest, tTest = read_dataset_to_X_and_y('dataset/Data-Test-mini.csv',normalization=normalizationMethod[i])

MSETest = linear_regression_evaluation(XTest, tTest, teta, printer=1)

print(f'vector learned parameters (θ0 , θ1 , ..., θn ) is\n{teta}\n')
print(f'MSE on train data is\n{MSETrain}\n')
print(f'MSE on test data is\n{MSETest}\n')
