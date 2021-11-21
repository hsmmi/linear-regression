from linearRegression import linear_regression, linear_regression_evaluation
from preprocessing import read_dataset_to_X_and_y


normalizationMethod = [None ,"z_score", "scaling"]
minValue, maxValue = 0, 1

for i in range(len(normalizationMethod)):
    print(f'Normalization method is {normalizationMethod[i]}\n')

    XTrain, tTrain = read_dataset_to_X_and_y('dataset/Data-Train.csv',normalization=normalizationMethod[i], minValue=minValue, maxValue=maxValue)

    teta, MSETrain = linear_regression(XTrain, tTrain , method='closed_form')
   
    XTest, tTest = read_dataset_to_X_and_y('dataset/Data-Test.csv',normalization=normalizationMethod[i], minValue=minValue, maxValue=maxValue)

    MSETest = linear_regression_evaluation(XTest, tTest, teta, plotter = 1)

    print(f'vector learned parameters (θ0 , θ1 , ..., θn ) is\n{teta}\n')
    print(f'MSE on train data is\n{MSETrain}\n')
    print(f'MSE on test data is\n{MSETest}\n')

    print(f'\n\n{"_"*50}\n\n')