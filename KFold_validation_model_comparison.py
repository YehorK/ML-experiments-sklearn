from sklearn.tree import DecisionTreeClassifier
from KFolds import k_fold_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier

##################################################
def comparison_of_models(x_combined,x_test,y_combined,y_test):
    #"""    
    #code with a for loop for checking accuracies of models
    nEstimators  = []
    AdaBoost_accuracy = []
    random_forest_accuracy = []

    for i in range (10, 81, 1):

        #Random Forest
        d = 57 # number of features
        d = math.isqrt(d)
        n_samples = len(x_combined)
        RF_model = RandomForestClassifier(
            criterion='gini',
            n_estimators = i,
            max_features = d,
            #random_state = 42,
            max_samples = n_samples,
            )
        RF_model.fit(x_combined, y_combined)
        RF_test_score = RF_model.score(x_test, y_test)

        #AdaBoost
        Adaboost_model = AdaBoostClassifier(
            n_estimators = i,
            #random_state = 42,
            estimator = DecisionTreeClassifier(
                max_depth = 1,
                ),
            learning_rate = 1,
            )    
        Adaboost_model.fit(x_combined, y_combined)
        AdaBoost_test_score = Adaboost_model.score(x_test, y_test)

        nEstimators.append(i)
        AdaBoost_accuracy.append(AdaBoost_test_score)
        random_forest_accuracy.append(RF_test_score)
    
    plt.plot(nEstimators,random_forest_accuracy)
    plt.plot(nEstimators,AdaBoost_accuracy)
    plt.grid()
    plt.legend(["Random Forest","AdaBoost"])
    plt.xlabel('Forest size') #Ensemble size
    plt.ylabel('Accuracy ')
    plt.show()
    #"""
"""
    d = 57 # number of features
    d = math.isqrt(d)
    n_samples = len(x_combined)
    RF_model = RandomForestClassifier(
        criterion='gini',
        n_estimators = 50,
        max_features = d,
        #random_state = 42,
        max_samples = n_samples,
        )
    RF_model.fit(x_combined, y_combined)
    RF_test_score = RF_model.score(x_test, y_test)
    print("Random Forest accuracy: ", RF_test_score)

    Adaboost_model = AdaBoostClassifier(
        n_estimators = 70,
        #random_state = 42,
        estimator = DecisionTreeClassifier(
            max_depth = 1,
            ),
        learning_rate = 2,
        )    
    Adaboost_model.fit(x_combined, y_combined)
    AdaBoost_test_score = Adaboost_model.score(x_test, y_test)
    print("AdaBoost accuracy: ", AdaBoost_test_score)
"""

##################################################

def Adaboost(x_combined,y_combined):

    #x_train,x_val,y_train,y_val = train_test_split(x_combined, y_combined, test_size=0.2, random_state=0)

    estimators_arr = []
    kfolds_mean = []

    for i in range (40, 202, 2):
        model_Adaboost = AdaBoostClassifier(
            n_estimators = i,
            random_state = 42,
            estimator = DecisionTreeClassifier(max_depth = 1),
            learning_rate = 1,
            )    
        #model_Adaboost.fit(x_train, y_train)
        combined_array = np.column_stack((x_combined, y_combined)) 
        KFolds_mean, KFold_std  = k_fold_validation(model_Adaboost, combined_array, 5)
        print(KFold_std)
        kfolds_mean.append(KFolds_mean)
        estimators_arr.append(i)
    
    plt.plot(estimators_arr,kfolds_mean)
    plt.grid()
    plt.legend(["Accuracy with K-fold"])
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy ')
    plt.title('Forest size vs Accuracy using K-fold')
    plt.show()


############################################
def randomForest(x_combined,y_combined):

    x_train,x_val,y_train,y_val = train_test_split(x_combined, y_combined, test_size=0.2, random_state=0)

    estimators_arr = []
    kfolds_mean = []

    d = 57 # number of features
    d = math.isqrt(d)
    n_samples = len(x_train)

    for i in range (10, 50, 5):
        model = RandomForestClassifier(
            criterion='gini',
            n_estimators = i,
            max_features = d,
            #random_state = 42,
            max_samples = n_samples,
            )
        #model.fit(x_train, y_train)
        combined_array = np.column_stack((x_combined, y_combined)) 
        KFolds_mean, KFold_std  = k_fold_validation(model, combined_array, 5)
        estimators_arr.append(i)
        kfolds_mean.append(KFolds_mean)

    plt.plot(estimators_arr,kfolds_mean)
    plt.grid()
    plt.legend(["Validation with K-fold"])
    plt.xlabel('Forest size')
    plt.ylabel('Accuracy ')
    plt.title('Forest size vs Accuracy using K-fold')
    plt.show()

##################################################
def main():

    input_data_file = np.loadtxt('spambase\spambase.data', delimiter=",")

    input_data_file = shuffle(input_data_file)

    x = (input_data_file[:,0:57])
    y = (input_data_file[:,-1])

    test_size_set = 0.2 #changed manually
    x_combined,x_test,y_combined,y_test = train_test_split(x, y, test_size=test_size_set, stratify=y, random_state=42)

    #randomForest(x_combined,y_combined)

    #Adaboost(x_combined,y_combined)
    
    comparison_of_models(x_combined,x_test,y_combined,y_test)

##################################################
# Calling main function
if __name__=="__main__":
    main()

