from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler


##################################################
def change_size_of_training_set(x, y):

    train_acc = []
    test_acc = []
    x_arr = []

    train_size_min = 0.05
    train_size_max = 0.9
    step = 0.05

    for i in np.arange(train_size_min, train_size_max, step):
        x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=i, test_size=1-i, random_state=42)

        model = DecisionTreeClassifier(
            random_state=42,
            #ccp_alpha=0.001
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
def pruning_and_corresponding_alphas(x_train, y_train, x_test, y_test):

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    path = model.cost_complexity_pruning_path(x_train, y_train)

    alphas = path['ccp_alphas']

    training_acc = []
    testing_acc = []

    for i in alphas:

        model = DecisionTreeClassifier(
            criterion = 'entropy',
            ccp_alpha=i,
            )

        model.fit(x_train, y_train)

        y_training_prediction = model.predict(x_train)
        y_testing_prediction = model.predict(x_test)

        training_acc.append(accuracy_score(y_train, y_training_prediction))
        testing_acc.append(accuracy_score(y_test, y_testing_prediction))

    y1 = training_acc
    y2 = testing_acc
    x = alphas
    plt.figure(figsize=(14,7))
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['train', 'test'], loc='upper right')
    plt.xlabel("Alphas")
    plt.ylabel("Accuracy")
    plt.title('Training vs Testing accuracy')
    plt.show()


##################################################
def gini_model(x_train, y_train, x_test, y_test):

    model_gini = DecisionTreeClassifier(criterion = "gini", 
                                        ccp_alpha = 0.001, 
                                        max_depth = 3, 
                                        random_state = 42,
                                        )
    model_gini.fit(x_train, y_train)
    print("Training accuracy: ", model_gini.score(x_train, y_train))
    print("Testing accuracy: ", model_gini.score(x_test, y_test))
    

##################################################
def DecisionStumpsModel(x, y):

    x_total_rows = len(x)
    x_total_cl = len(x[0])
    #temp = x[0:x_total_rows, x_total_cl-1]
    #print(temp)

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


    for i in range (0, x_total_cl):

        temp_X = x_train[0:x_total_rows, i]
        temp_X = temp_X.reshape(-1, 1)

        model = DecisionTreeClassifier(criterion = "gini", 
                                            ccp_alpha = 0.001, 
                                            max_depth = 10, 
                                            random_state = 42,
                                            )
        model.fit(temp_X, y_train)

        print(i, "Training accuracy: ", model.score(temp_X, y_train))

    print("\n")

    for i in range (0, x_total_cl):
        temp_X = x_test[0:x_total_rows, i]
        temp_X = temp_X.reshape(-1, 1)
        print(i, "Testing accuracy: ", model.score(temp_X, y_test))


##################################################
def entropy_model(x_train, y_train, x_test, y_test):

    model_entropy = DecisionTreeClassifier(criterion = "entropy", 
                                           ccp_alpha=0.001,
                                           max_depth = 3, 
                                           random_state = 42
                                           )
    model_entropy.fit(x_train, y_train)
    print("Training accuracy: ", model_entropy.score(x_train, y_train))
    print("Testing accuracy: ", model_entropy.score(x_test, y_test))


##################################################
def training_validation_comparison(x_train, y_train, x_test, y_test):

    maxDepth  = []
    trainAccuracy = []
    testingAccuracy = []
    for md in range(1,25):
        model = DecisionTreeClassifier(max_depth = md)
        model.fit(x_train, y_train)
        maxDepth.append(md)
        trainAccuracy.append(model.score(x_train,y_train))
        testingAccuracy.append(model.score(x_test,y_test))

    plt.plot(maxDepth,trainAccuracy)
    plt.plot(maxDepth,testingAccuracy)
    plt.grid()
    plt.legend(["Training","Validation"])
    plt.xlabel('Max Depth ')
    plt.ylabel('Accuracy ')
    plt.show()
    

################################################
def readDataFile():
    
    input_data_file = np.loadtxt('spambase\spambase.data', delimiter=",")

    input_data_file = shuffle(input_data_file)

    x = (input_data_file[:,0:57])
    y = (input_data_file[:,-1])

    return x, y


################################################
def main():

    x, y = readDataFile()

    """ Splitting dataset into training and testing """
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    """
    Please make sure to comment out functions below, which
    you do not need for easier and faster computation
    """

    #gini_model(x_train, y_train, x_test, y_test) #GINI

    #entropy_model(x_train, y_train, x_test, y_test) #Entropy

    #pruning_and_corresponding_alphas(x_train, y_train, x_test, y_test) #Pruning

    #training_validation_comparison(x_train, y_train, x_test, y_test) #Max Depth tuning

    change_size_of_training_set(x, y)

    #DecisionStumpsModel(x, y)

################################################
# Calling main function
if __name__=="__main__":
    main()