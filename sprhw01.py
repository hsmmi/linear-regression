from linearRegression import linearRegression, linearRegressionEvaluation
from preprocessing import readDatasetToXAndY


normalizationMethod = [None ,"z_score"]
# normalizationMethod = [None]
minValue, maxValue = 0, 1

for i in range(len(normalizationMethod)):
    print(f'Normalization method is {normalizationMethod[i]}\n')
    print(f'Linear regression with gradient decent alpha = 1\n')

    XTrain, yTrain = readDatasetToXAndY('dataset/Data-Train.csv',normalization=normalizationMethod[i], minValue=minValue, maxValue=maxValue)
    
    # theta, MSETrain = linearRegression(XTrain, yTrain, alpha= 1)
    theta, MSETrain = linearRegression(XTrain, yTrain, alpha= 1, ploter = 1)
   
    XTest, tTest = readDatasetToXAndY('dataset/Data-Test.csv',normalization=normalizationMethod[i], minValue=minValue, maxValue=maxValue)

    # MSETest = linearRegressionEvaluation(XTest, tTest, theta)
    MSETest = linearRegressionEvaluation(XTest, tTest, theta, plotter = 1)

    print(f'vector learned parameters (θ0 , θ1 , ..., θn ) is\n{theta}\n')
    print(f'MSE on train data is\n{MSETrain}\n')
    print(f'MSE on test data is\n{MSETest}\n')

    print(f'\n\n{"_"*50}\n\n')

for i in range(len(normalizationMethod)):
    print(f'Normalization method is {normalizationMethod[i]}\n')
    print(f'Linear regression with closed-form\n')

    XTrain, yTrain = readDatasetToXAndY('dataset/Data-Train.csv',normalization=normalizationMethod[i], minValue=minValue, maxValue=maxValue)
    
    theta, MSETrain = linearRegression(XTrain, yTrain)
   
    XTest, tTest = readDatasetToXAndY('dataset/Data-Test.csv',normalization=normalizationMethod[i], minValue=minValue, maxValue=maxValue)

    # MSETest = linearRegressionEvaluation(XTest, tTest, theta)
    MSETest = linearRegressionEvaluation(XTest, tTest, theta, plotter = 1)

    print(f'vector learned parameters (θ0 , θ1 , ..., θn ) is\n{theta}\n')
    print(f'MSE on train data is\n{MSETrain}\n')
    print(f'MSE on test data is\n{MSETest}\n')

    print(f'\n\n{"_"*50}\n\n')