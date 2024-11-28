import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

input_data_file = np.loadtxt('spambase\spambase.data', delimiter=",")

input_data_file = shuffle(input_data_file)

scaler = StandardScaler()

k = 10

folds = np.array_split(input_data_file, k)

modelType = RandomForestClassifier()

Accuracies = []

for i in range (k):

    arr_for_testing = folds[i]

    testing_targetArray = arr_for_testing[:,-1]
    testing_featuresArray = arr_for_testing[:,0:57]
    x_for_testing = np.array(testing_featuresArray)
    x_for_testing = scaler.fit_transform(x_for_testing)
    y_for_testing = np.array(testing_targetArray)

    training_array = np.concatenate((folds[:0] + folds[0+1:]), axis=0)
    training_targetArray = training_array[:,-1]
    training_featuresArray = training_array[:,0:57]
    x_for_training = np.array(training_featuresArray)
    x_for_training = scaler.fit_transform(x_for_training)
    y_for_training = np.array(training_targetArray)

    model = modelType
    model.fit(x_for_training, y_for_training)

    accuracy_testing = model.score(x_for_testing, y_for_testing)
    print("Accuracy: ", round(accuracy_testing*100, 2), "%")

    Accuracies.append(accuracy_testing)

KFolds_mean = np.mean(Accuracies)
KFold_std = np.std(Accuracies)

print(KFolds_mean)
print(KFold_std)