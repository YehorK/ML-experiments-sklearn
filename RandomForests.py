from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle


##################################################
def change_size_of_training_set(x, y):

    d = 57 # number of features
    d = math.isqrt(d)

    train_acc = []
    test_acc = []
    x_arr = []

    train_size_min = 0.1
    train_size_max = 0.9
    step = 0.05

    for i in np.arange(train_size_min, train_size_max, step):
        testing_size = 1 -i
        x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=i, test_size=testing_size, random_state=42)
        n_samples = len(x_train)
        model = RandomForestClassifier(
            random_state=42,
            max_features=d,
            max_samples=n_samples
            )
        model.fit(x_train, y_train)
        
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        train_acc.append(train_score)
        test_acc.append(test_score)
        x_arr.append(i)
    
    y1 = train_acc
    y2 = test_acc
    x = x_arr
    plt.figure(figsize=(14,7))
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['train', 'test'], loc='upper right')
    plt.xlabel("size of the training set")
    plt.ylabel("Accuracy")
    plt.title('Accuracies with varying training set')
    plt.show()


##################################################
def features_tuning(x, y):
    test_size_set = 0.2
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = test_size_set, stratify=y, random_state=42)

    #d = 57 # number of features
    #d = math.isqrt(d) 
    n_samples = math.floor(len(x) * (1 - test_size_set))   

    features  = []
    trainAccuracy = []
    testAccuracy = []
    for ftr in range (1, 15, 1):
        model = RandomForestClassifier(
            max_features = ftr, 
            random_state = 42, 
            max_samples = n_samples,
        )
        model.fit(x_train, y_train)
        features.append(ftr)
        trainAccuracy.append(model.score(x_train,y_train))
        testAccuracy.append(model.score(x_test,y_test))

    plt.plot(features,trainAccuracy)
    plt.plot(features,testAccuracy)
    plt.grid()
    plt.legend(["Training","Validation"],  loc='upper right')
    plt.xlabel('Number of features ')
    plt.ylabel('Accuracy ')
    plt.show()


##################################################
def maxD_tuning(x, y):
    
    test_size_set = 0.4
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = test_size_set, stratify=y, random_state=42)

    d = 57 # number of features
    d = math.isqrt(d) 
    n_samples = math.floor(len(x) * (1 - test_size_set))   

    # Check for different values of max depth
    maxDepth  = []
    trainAccuracy = []
    testAccuracy = []
    for maxD in range (1, 30):
        model = RandomForestClassifier(
            max_depth = maxD,
            max_features = d, 
            random_state = 42, 
            max_samples = n_samples,
        )
        model.fit(x_train, y_train)
        maxDepth.append(maxD)
        trainAccuracy.append(model.score(x_train,y_train))
        testAccuracy.append(model.score(x_test,y_test))

    plt.plot(maxDepth,trainAccuracy)
    plt.plot(maxDepth,testAccuracy)
    plt.grid()
    plt.legend(["Training","Validation"])
    plt.xlabel('Max Depth ')
    plt.ylabel('Accuracy ')
    plt.title('Max Depth vs Accuracy')
    plt.show()


##################################################
def estimators_tuning(x, y):

    test_size_set = 0.2
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = test_size_set, stratify=y, random_state=42)

    d = 57 # number of features
    d = math.isqrt(d)
    n_samples = math.floor(len(x) * (1 - test_size_set))   
    # Estimators check
    nEstimators  = []
    trainAccuracy = []
    valAccuracy = []
    for ne in range(1,120,5):
        model = RandomForestClassifier(
            n_estimators = ne,
            #random_state=42,
            max_samples = n_samples,
            max_features = 50,            
            )
        model.fit(x_train, y_train)
        nEstimators.append(ne)
        trainAccuracy.append(model.score(x_train,y_train))
        valAccuracy.append(model.score(x_test,y_test))

    plt.plot(nEstimators,trainAccuracy)
    plt.plot(nEstimators,valAccuracy)
    plt.grid()
    plt.legend(["Training","Validation"])
    plt.xlabel('Forest size') #Ensemble size
    plt.ylabel('Accuracy ')
    plt.title('Forest size vs Accuracy')
    plt.show()


##################################################
def grid_search_function(x, y):

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state=42)

    # scaler = StandardScaler()
    # x_test = scaler.fit_transform(x_test)
    # x_train = scaler.fit_transform(x_train)

    parameters_for_testing = {
    'n_estimators':[5,10,15,20,25,30,35,40,45,50,55,60,65,70,80,85,90,95,100],
    'max_depth':[5,7,9,11,13,15,17,20,25,30]
    }

    gridSearch_model = RandomForestClassifier(
        random_state=42,
        )

    grid_search = GridSearchCV(
        gridSearch_model, #our model
        param_grid = parameters_for_testing, #parameters to test
        cv = 5,  #number of folds
        )
        
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_ 
    print("BEST PARAMS:")
    print(best_params)

    RF_gridSearch_BestModel = RandomForestClassifier(**best_params)
    RF_gridSearch_BestModel.fit(x_train, y_train)

    print("Accuracy on Training",RF_gridSearch_BestModel.score(x_train,y_train))
    print("Accuracy on Test",RF_gridSearch_BestModel.score(x_test,y_test))

##################################################
def RandomForestSimple(x, y):
    test_size_set = 0.2 #changed manually

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=test_size_set, stratify=y, random_state=42)

    d = 57 # number of features
    d = math.isqrt(d)
    n_samples = len(x_train)

    model = RandomForestClassifier(
        n_estimators = 100,
        max_features = d,
        #random_state = 42, # the Answer to the Ultimate Question of Life, the Universe, and Everything
        max_samples = n_samples,
        #ccp_alpha=0.001,
        #max_depth = 5,
        )
    model.fit(x_train, y_train)

    print("Accuracy on Training",model.score(x_train,y_train))
    print("Accuracy on Test",model.score(x_test,y_test))


##################################################
def readDataFile():
    
    input_data_file = np.loadtxt('spambase\spambase.data', delimiter=",")

    input_data_file = shuffle(input_data_file)

    x = (input_data_file[:,0:57])
    y = (input_data_file[:,-1])


    return x, y


##################################################
def main():

    x, y = readDataFile()

    """
    Please make sure to comment out functions below, which
    you do not need for easier and faster computation
    """

    #RandomForestSimple(x, y) #Simple random forests implementation (using RFclassifier)
    
    #grid_search_function(x, y) #Implementation of the GridSearchCV for finding the best parameters

    #maxD_tuning(x, y)

    #estimators_tuning(x,y)

    #features_tuning(x, y)

    change_size_of_training_set(x, y)

##################################################
# Calling main function
if __name__=="__main__":
    main()