import numpy as np
from matplotlib import pyplot as plt

def closed_form(XInput,yInput):
    """
    It gets matrix samples(XInput) any vector lables(yInput)
    Compute learning parameters with (((X)tX)^-1)(X)ty
    Return learned parameters
    """
    return np.linalg.inv(XInput.transpose()@XInput)@XInput.transpose()@yInput

def linear_regression(XTrain, yTrain, method, printer = 0):
    """
    It gets matrix samples train(XTrain) and their labels(yTrain) and method.
    method:
    .   can be "closed_form" or "gradient_descent"
    Return learned parameters (θ0 , θ1 , ..., θn ) and the value of MSE 
    error on the train data.
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

def linear_regression_evaluation(XTest, yTest, teta, printer = 0, plotter=0):
    """
    It gets matrix samples test(XTest) and their labels(yTest) and learned
    parameters(teta) and plotter(by default 0).
    If plotter be 1 then it'll plot the test sample and a regression line
    Return the value of MSE error on the test data.
    """
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

    if(plotter):
        sXTest = XTest.argmin(axis=0)[1]
        eXTest = XTest.argmax(axis=0)[1]

        plt.plot(list(zip(*XTest))[1], yTest, ".", label="samples")
        plt.plot([XTest[sXTest][1],XTest[eXTest][1]], [predictionTest[sXTest],predictionTest[eXTest]], "-r", label="regression line")
        plt.legend(loc="upper left")
        plt.xlabel('Feature')
        plt.ylabel('Lable')
        plt.show()

    return MSETest

