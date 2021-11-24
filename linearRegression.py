import numpy as np
from matplotlib import pyplot as plt

def closed_form(XInput,yInput):
    """
    It gets matrix samples(XInput) and vector labels(yInput)
    Compute learning parameters with (((X)tX)^-1)(X)ty
    Return learned parameters
    """
    return np.linalg.inv(XInput.T@XInput)@XInput.T@yInput

def gradient_descent(XInput ,yInput, alpha, printer, ploter):
    """
    It gets matrix samples(XInput) and vector labels(yInput)
    Compute learning parameters with updating all theta(i)
    in each epochs and stop if #epochs exit the 
    threshold or cost function improvment < eps
    Return learned parameters
    """

    def MSE(XInput ,yInput, theta):
        return ((XInput @ theta - yInput).T@(XInput @ theta - yInput))[0][0] / len(XInput)

    def updateTheta(XInput ,yInput, theta, alpha):
        return theta - alpha * (XInput.T @ ((XInput @ theta) - yInput))
    
    def step_decay(epoch,epochs_drop):
        """
        It half the learning rate every epochs_drop epochs
        """
        drop = 0.5
        newAlpha = alpha * drop**((1+epoch)//epochs_drop)
        return newAlpha

    theta = np.random.rand(len(XInput[0]),1)
    maxEpoch = int(1e5)
    J = []
    epoch = 0
    eps = 1e-3
    alpha /= len(XInput)

    for i in range(1,maxEpoch):
        theta = updateTheta(XInput ,yInput, theta, step_decay(i,20))
        epoch += 1
        J.append(MSE(XInput ,yInput, theta))
        if(i > 1 and abs(J[-2]-J[-1]) < eps):
            break
    
    if(printer):
        print(f'MSE in each epochs are \n{J}')
        print(f'After {epoch} epochs')
  
    if(ploter):
        plt.plot(range(1,len(J)+1), J, ".--", label="cost function")
        plt.legend(loc="upper right")
        plt.xlabel('Iteratioin')
        plt.ylabel('J(θ)')
        plt.show()
    
    return theta

def linear_regression(XTrain, yTrain, alpha = None, printer = 0, ploter = 1):
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
        theta = gradient_descent(XTrain,yTrain,alpha,printer,ploter)

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
        plt.legend(loc="upper right")
        plt.xlabel('Feature')
        plt.ylabel('Lable')
        plt.show()

    return MSETest

