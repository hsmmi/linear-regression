import numpy as np
from matplotlib import pyplot as plt

from preprocessing import read_dataset_to_X_and_y

def closed_form(XInput,yInput):
    """
    It gets matrix samples(XInput) and vector lables(yInput)
    Compute learning parameters with (((X)tX)^-1)(X)ty
    Return learned parameters
    """
    return np.linalg.inv(XInput.T@XInput)@XInput.T@yInput

def gradient_descent(XInput ,yInput, alpha):
    """
    It gets matrix samples(XInput) and vector lables(yInput)
    Compute learning parameters with updating all theta(i)
    in each iteration and stop if #iteration exit the 
    threshold or theta converge
    Return learned parameters
    """

    thetaNew = np.zeros((len(XInput[0]),1))
    theta = np.ones_like(thetaNew)
    SSDT = 1
    maxIteration = len(XInput)*10
    iteration = 0
    eps = 1e-9
    # while (iteration < maxIteration and SSDT > eps):
    while (SSDT > eps):
        thetaNew = theta - alpha * (XInput.T @ ((XInput @ theta) - yInput))
        
        SSDT = np.linalg.norm(thetaNew - theta)
        theta = thetaNew
        iteration += 1
    print(iteration)
    return theta

def linear_regression(XTrain, yTrain, alpha = None, printer = 0):
    """
    It gets matrix samples train(XTrain) and their labels(yTrain) and method.
    alpha:
    .   If alpha be None it will use closed_form if not it use gradient_descent
        with your alpha
    Return learned parameters (θ0 , θ1 , ..., θn ) and the value of MSE 
    error on the train data.
    """
    if(printer):
        print(f'matrix XTrain is\n{XTrain}\n')
    if(printer):
        print(f'vector yTrain is\n{yTrain}\n')

    if(alpha == None):
        theta = closed_form(XTrain,yTrain)
    else:
        theta = gradient_descent(XTrain,yTrain,alpha)

    if(printer):
        print(f'vector learned parameters (θ0 , θ1 , ..., θn ) is\n{theta}\n')

    predictionTrain = XTrain @ theta
    if(printer):
        print(f'vector predictionTrain is\n{predictionTrain}\n')

    errorTrain = predictionTrain - yTrain
    if(printer):
        print(f'vector errorTrain is\n{errorTrain}\n')

    MSETrain = (np.square(errorTrain)).mean(axis=0)
    if(printer):
        print(f'MSE on train data is\n{MSETrain}\n')
    
    return theta, MSETrain

def linear_regression_evaluation(XTest, yTest, theta, printer = 0, plotter=0):
    """
    It gets matrix samples test(XTest) and their labels(yTest) and learned
    parameters(theta) and plotter(by default 0).
    If plotter be 1 then it'll plot the test sample and a regression line
    Return the value of MSE error on the test data.
    """
    if(printer):
        print(f'matrix XTest is\n{XTest}\n')
    if(printer):
        print(f'vector yTest is\n{yTest}\n')

    predictionTest = XTest @ theta
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

