# ML-experiments-sklearn

The implementation was originally developed as an assignment for the data mining course.

The instructions below describe the provided python files and the functions they contain. This repository contains multiple models for dealing with a binary classification task.

The data for models tuning and experimentation is spambase dataset from UCI: https://archive.ics.uci.edu/dataset/94/spambase

Note:
- All written code is in Python. All ML algorithms are implemented using scikit-learn library.
- The naming of each file stands for what algorithm that's implmented in there.
- Every file has a function main(), which is primarily executed. From there you can call other functions, for example, a function where a specific hyperparameter is tuned, etc. You may comment out/uncomment functions in the main() in order to test a specific function, spending less time on computing.
- In several files you may find commented out different parts of the code. You may try to uncomment and see how it affects the rest of the code. Also, throughout the code almost no scaling (pre-processing) was used as the models were able to get quite good result without it.
- There are some comments in every file in order to assist with understanding the code logic.
- Files are not connected to one another (except KFolds.py). That means in order to test/tune Random Forest, you would need to open only the RandomForests.py file.... same goes for other files.

--------------------------------------------

# DecisionTrees.py

The main() in this file contained the following functions:
- gini_model (decision trees with Gini split)
- entropy_model (decision trees with Entropy split)
- pruning_and_corresponding_alphas (Pruning)
- training_validation_comparison (experimenting with changing Max Depth)
- change_size_of_training_set

Make sure that all but one of the above mentioned fns are commented out in order to make computation time smaller. You may comment/uncomment the functions one by one to check individual performances

There is also a function readDataFile() - it reads the data from data set and divides it on x and y

--------------------------------------------

# RandomForests.py

The main() in this file contained the following functions:
- RandomForestSimple (just a simple implmentation of Random Forest using sklearn)
- maxD_tuning (tuning max depth)
- estimators_tuning (tuning the forest size)
- features_tuning (tuning # of features)
- change_size_of_training_set
- grid_search_function (tried to use and see the outcome of the CV grid search fnc by sklearn that finds the most optimal parameters for the model)

There is also a function readDataFile() - it reads the data from data set and divides it on x and y

--------------------------------------------
# AdaBoost

The main() in this file contained the following functions:
- AdaBoostAlgorithmDefault
- tuningMaxDepth
- tuning_n_estimators
- tuningLearningRate
- change_size_of_training_set

There is also a function readDataFile() - it reads the data from data set and divides it on x and y

--------------------------------------------

# KFolds.py

Similar to KFolds_plain_code.py, in fact the code is the same, but KFolds.py is organized as a function that can be called from other files. It returns the mean of accuracies for k-fold validation, and the standard deviation.

--------------------------------------------

# KFolds_plain_code.py

This is a file similar to KFolds.py, however, the code here is written in a "plain" manner, i.e. there are no functions, no main(), just some plain code. It was very helpful in hard coding the k-fold cross validation, and it can be executed without interacting with any other code/files.

--------------------------------------------

# KFolds_Validation_model_comparison.py

AdaBoost and Random Forest models are compared in this file. K-fold validfation is used for estimate the mean error for each model for different forest sizes.

The main() in this file contained the following functions:
- randomForest
- Adaboost
- comparison_of_models

  
