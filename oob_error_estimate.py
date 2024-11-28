from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state  
from sklearn.utils import shuffle
import numpy as np

input_data_file = np.loadtxt('spambase\spambase.data', delimiter=",")
input_data_file = shuffle(input_data_file)

x = (input_data_file[:,0:57])
y = (input_data_file[:,-1])

test_size_set = 0.2
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = test_size_set, stratify=y, random_state=42)

M = 151

oob_error_estimate_sklearn = [] #done by sklearn

oob_mean_computed = [] #computed (hard coded)

forest = []

for i in range(10, M, 1):

    rf = RandomForestClassifier(
        #warm_start = True, 
        n_estimators = i, 
        max_features='sqrt',
        random_state = 42,
        oob_score = True
        )
    rf = rf.fit(x_train, y_train)
    oob_error = 1 - rf.oob_score_ 

    forest.append(i)
    oob_error_estimate_sklearn.append(oob_error)

    n_samples = len(x_train)
    n_samples_bootstrap = n_samples

    unsampled_indices_for_all_trees= []
    for estimator in rf.estimators_: #the code in this for loop has been prepared by Prof. Mehta
        random_instance = check_random_state(estimator.random_state)
        sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
        sample_counts = np.bincount(sample_indices, minlength = n_samples)
        unsampled_mask = sample_counts == 0
        indices_range = np.arange(n_samples)
        unsampled_indices = indices_range[unsampled_mask]
        unsampled_indices_for_all_trees += [unsampled_indices]

    my_oob_error_arr = []

    for j in range (0, i, 1):

        temp = unsampled_indices_for_all_trees[j]

        rows = len(temp)
        cols = 57
        x_new = np.zeros((rows, cols))
        y_new = np.zeros((rows))

        for n in range (0, rows):
            x_new[n, 0:cols]  = x_train[temp[n], 0:cols]
            y_new[n] = y_train[temp[n]]

        obb_validation_score = rf.score(x_new, y_new)
        oob_error_computed = 1- obb_validation_score
        my_oob_error_arr.append(oob_error_computed)

    oob_computed_mean = np.mean(my_oob_error_arr)
    oob_mean_computed.append(oob_computed_mean)


plt.plot(forest, oob_error_estimate_sklearn)
plt.grid()
#plt.legend(["OOB error"])
plt.xlabel('Forest size') #Ensemble size
plt.ylabel('OOB error')
plt.title('Forest size vs Accuracy')
plt.show()


plt.plot(forest, oob_mean_computed)
plt.grid()
#plt.legend(["OOB error"])
plt.xlabel('Forest size') #Ensemble size
plt.ylabel('OOB error')
plt.title('Forest size vs Accuracy')
plt.show()