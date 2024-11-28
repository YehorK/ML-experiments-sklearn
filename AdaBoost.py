import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from KFolds import k_fold_validation
from sklearn.preprocessing import StandardScaler


##################################################
def readDataFile():
    
    input_data_file = np.loadtxt('spambase\spambase.data', delimiter=",")

    input_data_file = shuffle(input_data_file)

    x = (input_data_file[:,0:57])
    y = (input_data_file[:,-1])

    return x, y


##################################################
def change_size_of_training_set(x, y):

    train_acc = []
    test_acc = []
    x_arr = []

    train_size_min = 0.1
    train_size_max = 0.8
    step = 0.01

    for i in np.arange(train_size_min, train_size_max, step):
        testing_size = 1 -i
        x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=i, test_size=testing_size, random_state=42)
        model = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=0.3,
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
def AdaBoostAlgorithmDefault (x, y):

    test_size_set = 0.2 #changed manually

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=test_size_set, random_state = 42)

    # scaler = StandardScaler()
    # x_test = scaler.fit_transform(x_test)
    # x_train = scaler.fit_transform(x_train)

    # The algorithm with default settings
    # Different parameters can be altered to see how it effects accuracy
    model_Adaboost = AdaBoostClassifier(
        #n_estimators = 50, #number of weak learners
        #random_state = 42,
        estimator = DecisionTreeClassifier(max_depth = 1),
        learning_rate = 1,
        )
    
    model_Adaboost.fit(x_train, y_train)

    y_pred = model_Adaboost.predict(x_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


##################################################
def tuningMaxDepth(x, y):
    
    test_size_set = 0.2 #changed manually

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=test_size_set, random_state = 42)

    # scaler = StandardScaler()
    # x_test = scaler.fit_transform(x_test)
    # x_train = scaler.fit_transform(x_train)

    # TUNING MAX DEPTH
    maxDepth  = []
    trainAccuracy = []
    testAccuracy = []
    for maxD in range (1, 11):
        model = AdaBoostClassifier(
            estimator = DecisionTreeClassifier(max_depth = maxD),
            n_estimators = 50, 
            learning_rate = 1,
        )
        model.fit(x_train, y_train)
        maxDepth.append(maxD)
        trainAccuracy.append(model.score(x_train,y_train))
        testAccuracy.append(model.score(x_test,y_test))

    plt.plot(maxDepth,trainAccuracy)
    plt.plot(maxDepth,testAccuracy)
    plt.grid()
    plt.legend(["Training","Validation"], loc='upper right')
    plt.xlabel('Max Depth ')
    plt.ylabel('Accuracy ')
    plt.title("Max Depth vs Accuracy - AdaBoost")
    plt.show()


##################################################
def tuning_n_estimators(x,y):

    test_size_set = 0.8 #changed manually

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=test_size_set, random_state = 42)

    # scaler = StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    # TUNNING N_ESTIMATOR 
    max_n_estimators  = []
    trainAccuracy = []
    testAccuracy = []
    for maxNEstim in range (1, 100, 5):
        model = AdaBoostClassifier(
            estimator = DecisionTreeClassifier(max_depth = 10),
            n_estimators = maxNEstim, 
            learning_rate = 1,
        )
        model.fit(x_train, y_train)
        max_n_estimators.append(maxNEstim)
        trainAccuracy.append(model.score(x_train,y_train))
        testAccuracy.append(model.score(x_test,y_test))

    plt.plot(max_n_estimators,trainAccuracy)
    plt.plot(max_n_estimators,testAccuracy)
    plt.grid()
    plt.legend(["Training","Validation"], loc='upper right')
    plt.xlabel('n_estimators ')
    plt.ylabel('Accuracy ')
    plt.title("Impact of number of weak learners on accuracy")
    plt.show()


##################################################
def tuningLearningRate (x, y):

    test_size_set = 0.2 #changed manually

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=test_size_set, random_state = 42)

    # TUNNING LEARNING RATE 
    max_LR  = []
    trainAccuracy = []
    testAccuracy = []
    for lr in np.arange(0.001, 1.2, 0.005):
        model = AdaBoostClassifier(
            #estimator = DecisionTreeClassifier(max_depth = 1),
            n_estimators = 50, 
            #random_state=42,
            learning_rate = lr,
        )
        model.fit(x_train, y_train)
        max_LR.append(lr)
        trainAccuracy.append(model.score(x_train,y_train))
        testAccuracy.append(model.score(x_test,y_test))

    plt.plot(max_LR,trainAccuracy)
    plt.plot(max_LR,testAccuracy)
    plt.grid()
    plt.legend(["Training","Validation"])
    plt.xlabel('Learning rate ')
    plt.ylabel('Accuracy ')
    plt.show()
    

##################################################
def main():

    # Read data from the dataset file
    x, y = readDataFile()

    """
    Please make sure to comment out functions below, which
    you do not need for easier and faster computation
    """

    #AdaBoostAlgorithmDefault(x, y) 

    #tuningMaxDepth(x, y)

    #tuning_n_estimators(x, y)

    #tuningLearningRate(x, y)

    change_size_of_training_set(x,y)

##################################################
# Calling main function
if __name__=="__main__":
    main()