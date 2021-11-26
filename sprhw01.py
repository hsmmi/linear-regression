from linear_regression import linear_regression, linear_regression_evaluation
from preprocessing import read_dataset_to_X_and_y

normalization_method = [None ,"z_score"]
min_value, max_value = 0, 1

for i in range(len(normalization_method)):
    print(f'Normalization method is {normalization_method[i]}\n')
    print(f'Linear regression with gradient decent alpha = 0.3\n')

    X_train, y_train = read_dataset_to_X_and_y('dataset/Data-Train.csv',normalization=normalization_method[i], min_value=min_value, max_value=max_value)
    
    # theta, MSE_train = linear_regression(X_train, y_train, alpha= 1)
    theta, MSE_train = linear_regression(X_train, y_train, alpha = 0.3, plotter = 1)
   
    X_test, y_test = read_dataset_to_X_and_y('dataset/Data-Test.csv',normalization=normalization_method[i], min_value=min_value, max_value=max_value)

    # MSE_test = linear_regression_evaluation(X_test, y_test, theta)
    MSE_test = linear_regression_evaluation(X_test, y_test, theta, plotter = 1)

    print(f'vector learned parameters (θ0 , θ1 , ..., θn) is\n{theta}\n')
    print(f'MSE on train data is\n{MSE_train}\n')
    print(f'MSE on test data is\n{MSE_test}\n')

    print(f'\n\n{"_"*50}\n\n')

for i in range(len(normalization_method)):
    print(f'Normalization method is {normalization_method[i]}\n')
    print(f'Linear regression with closed-form\n')

    X_train, y_train = read_dataset_to_X_and_y('dataset/Data-Train.csv',normalization=normalization_method[i], min_value=min_value, max_value=max_value)
    
    theta, MSE_train = linear_regression(X_train, y_train)
   
    X_test, y_test = read_dataset_to_X_and_y('dataset/Data-Test.csv',normalization=normalization_method[i], min_value=min_value, max_value=max_value)

    # MSE_test = linear_regression_evaluation(X_test, y_test, theta)
    MSE_test = linear_regression_evaluation(X_test, y_test, theta, plotter = 1)

    print(f'vector learned parameters (θ0 , θ1 , ..., θn) is\n{theta}\n')
    print(f'MSE on train data is\n{MSE_train}\n')
    print(f'MSE on test data is\n{MSE_test}\n')

    print(f'\n\n{"_"*50}\n\n')