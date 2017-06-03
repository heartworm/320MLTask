import csv
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
import sklearn as sk
import matplotlib.pyplot as plt
'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls different functions to perform the required tasks.

'''




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9286675, 'Shravan', 'Lal') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    X_pylist = []
    y_pylist = []

    with open(dataset_path, newline='') as data_file:
        data_reader = csv.reader(data_file)
        for row in data_reader:
            X_pylist.append([float(num) for num in row[2:]]) #slice from 2nd index
            y_pylist.append(1 if row[1] == 'M' else 0) #1 if malignant else 0


    return np.array(X_pylist), np.array(y_pylist)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training, parameter=None): #third arg to conform to the same scheme as others
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
#    raise NotImplementedError()

    classifier = GaussianNB()
    classifier.fit(X_training, y_training)
    return classifier

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training,  max_depth=None):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Decision tree has random components in algorithm, seed to 123456 as well
    classifier = DecisionTreeClassifier(max_depth = max_depth, random_state=123456)
    classifier.fit(X_training, y_training)
    return classifier
    
def cv_classifier(X, y, clf_builder, param_values): # perform cross validation on a classifier
    '''
    Perform Cross validation of an SKLearn classifier given several hyperparameter values, returning training
    and test accuracy for each value. 
    
    :param X: ndarray with features, with each row being a training point
    :param y: ndarray with target categories corresponding to rows of X
    :param clf_builder: function that returns an SKLearn classifier given arguments (X, Y, parameter_value) 
    :param param_values: sequence of numbers to test as parameter_value passed into clf_builder. 
    :return (performance in training set, performance in validation set): both lists, where list[index] is the accuracy of 
            the classifier built using param_values[index]
    '''


    perf_training = []
    perf_validation = []

    for param_value in param_values: #cycle through parameter values to test
        cur_perf_training = 0 #sum of accuracies to be divided later on, to get average performance over KFolds
        cur_perf_validation = 0

        num_splits = 3
        kf = KFold(n_splits=num_splits)

        for ind_training, ind_validation in kf.split(X):
            X_training = X[ind_training]
            X_validation = X[ind_validation]
            y_training = y[ind_training]
            y_validation = y[ind_validation]

            # Create the classifier with selected parameter
            classifier = clf_builder(X_training, y_training, param_value)
            cur_perf_training += get_accuracy(classifier, X_training, y_training)
            cur_perf_validation += get_accuracy(classifier, X_validation, y_validation)

        perf_training.append(cur_perf_training / num_splits)
        perf_validation.append(cur_perf_validation / num_splits)

    return perf_training, perf_validation

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training, n_neighbors = 5):
    ''' 
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''

    # weights = 'uniform' or 'distance' to prioritise neighbours votes based on proximity
    classifier = KNeighborsClassifier(n_neighbors = n_neighbors, weights = 'uniform')
    classifier.fit(X_training, y_training)
    return classifier
    
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training, C_value=1.0):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''

    classifier = SVC(kernel='linear', C=C_value)
    classifier.fit(X_training, y_training)
    return classifier
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_accuracy(classifier, X_testing, y_testing):
    '''
    Return accuracy score for an SKLearn classifier object upon a dataset
    :param classifier: SKLearn classifier object
    :param X_testing: Feature matrix
    :param y_testing: Correct classifications
    :return: Float number from 0 to 1 representing accuracy of the classifier
    '''
    pred_testing = classifier.predict(X_testing)
    return sk.metrics.accuracy_score(y_testing, pred_testing)


if __name__ == "__main__":
    TESTING_SETASIDE = 0.1 # Set aside 10% of the set for final testing.

    X, y = prepare_dataset("medical_records.data") # Get the X and y datasets
    X_training, X_testing, y_training, y_testing = sk.model_selection.train_test_split(X, y, test_size=TESTING_SETASIDE,
                                                                                       random_state=123456)

    # Building graph for Nearest Neighbors Parameters -------------------------------------------
    # Lists to build a graph of training vs validation accuracy as the chosen hyperparameter changes.
    # Accuracy values as floats from 0-1, these are heights of the graph points
    perf_NN_values = list(range(1,26)) # Tested values of the parameter, displayed on the X axis
    perf_NN_training, perf_NN_validation = cv_classifier(X_training, y_training, build_NN_classifier, perf_NN_values)

    fig_NN = plt.figure() # Create a figure window
    plt.plot(perf_NN_values, perf_NN_training, 'b.--', label="Training Set Accuracy")
    plt.plot(perf_NN_values, perf_NN_validation, 'r.--', label="Validation Set Accuracy")
    plt.title("Cross-Validation of K-Nearest Neighbors Classifier")
    plt.xlabel("Number of Considered Neighbors (K)")
    plt.ylabel("Accuracy")
    plt.xticks(perf_NN_values)
    plt.legend()

    # Building graph for Nearest Neighbors Parameters -------------------------------------------
    # Building graph for parameters for Decision Tree Classifier. See Nearest Neighbors graph for explanation

    perf_DT_values = list(range(1,11))
    perf_DT_training, perf_DT_validation = cv_classifier(X_training, y_training, build_DT_classifier, perf_DT_values)
    fig_DT = plt.figure()
    plt.plot(perf_DT_values, perf_DT_training, 'b.--', label="Training Set Accuracy")
    plt.plot(perf_DT_values, perf_DT_validation, 'r.--', label="Validation Set Accuracy")
    plt.title("Cross-Validation of Decision Tree Classifier")
    plt.xlabel("Maximum Depth of Decision Tree")
    plt.ylabel("Accuracy")
    plt.xticks(perf_DT_values)
    plt.legend()

    # Building graph for parameters for SVM Classifier. See Nearest Neighbors graph for explanation

    perf_SVM_values = [2**i for i in range(11)]
    perf_SVM_training, perf_SVM_validation = cv_classifier(X_training, y_training, build_SVM_classifier, perf_SVM_values)
    fig_DT = plt.figure()
    plt.plot(perf_SVM_values, perf_SVM_training, 'b.--', label="Training Set Accuracy")
    plt.plot(perf_SVM_values, perf_SVM_validation, 'r.--', label="Validation Set Accuracy")
    plt.title("Cross-Validation of SVM Classifier")
    plt.xlabel("Value of Mis-classification Penalty Coefficient C")
    plt.ylabel("Accuracy")
    plt.xscale('log', basex=2)
    plt.xticks(perf_SVM_values)
    plt.legend()

    chosen_param_NN = 6 #k number of neighbors
    chosen_param_DT = 6 #maximum depth of tree
    chosen_param_SVM = 2**4 #C value

    classifier_NN = build_NN_classifier(X_training, y_training, chosen_param_NN)
    classifier_SVM = build_SVM_classifier(X_training, y_training, chosen_param_SVM)
    classifier_DT = build_DT_classifier(X_training, y_training, chosen_param_DT)
    classifier_NB = build_NB_classifier(X_training, y_training)

    print("Accuracy on Testing Data ------------------")
    print("Accuracy of NN", get_accuracy(classifier_NN, X_testing, y_testing))
    print("Accuracy of SVM", get_accuracy(classifier_SVM, X_testing, y_testing))
    print("Accuracy of DT", get_accuracy(classifier_DT, X_testing, y_testing))
    print("Accuracy of NB", get_accuracy(classifier_NB, X_testing, y_testing))

    plt.show()
