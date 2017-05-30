import csv
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
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

def build_NB_classifier(X_training, y_training):
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
    pred_testing = classifier.predict(X_testing)
    return sk.metrics.accuracy_score(y_testing, pred_testing)


if __name__ == "__main__":
    TRAINING_SETASIDE = 0.8 # Set aside 80% of the dataset to training
    VALIDATION_SETASIDE = 0.1 # use 10% for cross-validation and choosing hyperparameters

    X, y = prepare_dataset("medical_records.data") # Get the X and y datasets


    # Randomizing the dataset ----------------------
    np.random.seed(123451268) # Seed the RNG so that we get predictable results.
    dataset_mat = np.concatenate([X, y[:, np.newaxis]], 1) # Join X any Y together so that they can be shuffled as one
    np.random.shuffle(dataset_mat) # Shuffle the joined ndarray in place to remove any ordering effects
    X_shuffled = dataset_mat[:, 0:-1] # Split them into X and Y once more, noticing that the last column is the Y vector
    y_shuffled = dataset_mat[:, -1]



    num_points = X.shape[0] # The number of samples is the number of rows in the X vector
    num_training = math.floor(num_points * TRAINING_SETASIDE) # Convert the setaside fractions to actual sample counts
    num_validation = math.floor(num_points * VALIDATION_SETASIDE)

    # Set aside the first num_training for training, then the next num_validation for validation
    # then the rest for testing. Apply to both X and y
    X_training = X_shuffled[0:num_training]
    X_validation = X_shuffled[num_training:num_training + num_validation]
    X_testing = X_shuffled[num_training + num_validation:]

    y_training = y_shuffled[0:num_training]
    y_validation = y_shuffled[num_training:num_training+num_validation]
    y_testing = y_shuffled[num_training + num_validation:]

    # Building graph for Nearest Neighbors Parameters -------------------------------------------

    # Lists to build a graph of training vs validation accuracy as the chosen hyperparameter changes.
    perf_NN_validation = [] # Accuracy values as floats from 0-1, these are heights of the graph points
    perf_NN_training = [] # Same as perf_NN_training
    perf_NN_values = [] # Tested values of the parameter, displayed on the X axis

    fig_NN = plt.figure() # Create a figure window
    for n_neighbors in range(1,26): # Range encompasses tested values for the parameter
        # Create the classifier with selected parameter
        classifier = build_NN_classifier(X_training, y_training, n_neighbors=n_neighbors)
        # Add to the graph lists
        perf_NN_values.append(n_neighbors)
        perf_NN_training.append(get_accuracy(classifier, X_training, y_training))
        perf_NN_validation.append(get_accuracy(classifier, X_validation, y_validation))
    plt.plot(perf_NN_values, perf_NN_training, 'b.--', label="Training Set Accuracy")
    plt.plot(perf_NN_values, perf_NN_validation, 'r.--', label="Validation Set Accuracy")
    plt.title("Cross-Validation of K-Nearest Neighbors Classifier")
    plt.xlabel("Number of Considered Neighbors (K)")
    plt.ylabel("Accuracy")
    plt.xticks(range(1,26))
    plt.legend()


    # Building graph for parameters for Decision Tree Classifier. See Nearest Neighbors graph for explanation

    perf_DT_validation = []
    perf_DT_training = []
    perf_DT_values = []
    fig_DT = plt.figure()
    for max_depth in range(1, 11):
        classifier = build_DT_classifier(X_training, y_training, max_depth=max_depth)
        perf_DT_values.append(max_depth)
        perf_DT_training.append(get_accuracy(classifier, X_training, y_training))
        perf_DT_validation.append(get_accuracy(classifier, X_validation, y_validation))
    plt.plot(perf_DT_values, perf_DT_training, 'b.--', label="Training Set Accuracy")
    plt.plot(perf_DT_values, perf_DT_validation, 'r.--', label="Validation Set Accuracy")
    plt.title("Cross-Validation of Decision Tree Classifier")
    plt.xlabel("Maximum Depth of Decision Tree")
    plt.ylabel("Accuracy")
    plt.xticks(range(1,11))
    plt.legend()

    # Building graph for SVM parameters, see Nearest Neighbors graph for explanation

    perf_SVM_validation = []
    perf_SVM_training = []
    perf_SVM_values = []
    fig_SVM = plt.figure()
    for C_value in [2**i for i in range(11)]:
        classifier = build_SVM_classifier(X_training, y_training, C_value = C_value)
        perf_SVM_values.append(C_value)
        perf_SVM_training.append(get_accuracy(classifier, X_training, y_training))
        perf_SVM_validation.append(get_accuracy(classifier, X_validation, y_validation))
    plt.plot(perf_SVM_values, perf_SVM_training, 'b.--', label="Training Set Accuracy")
    plt.plot(perf_SVM_values, perf_SVM_validation, 'r.--', label="Validation Set Accuracy")
    plt.title("Cross-Validation of SVM Classifier")
    plt.xlabel("Value of Error Penalty Coefficient C")
    plt.ylabel("Accuracy")
    plt.legend()



    classifier_NN = build_NN_classifier(X_training, y_training)
    classifier_SVM = build_SVM_classifier(X_training, y_training)
    classifier_DT = build_DT_classifier(X_training, y_training)
    classifier_NB = build_NB_classifier(X_training, y_training)

    print("Accuracy on Testing Data ------------------")
    print("Accuracy of NN", get_accuracy(classifier_NN, X_testing, y_testing))
    print("Accuracy of SVM", get_accuracy(classifier_SVM, X_testing, y_testing))
    print("Accuracy of DT", get_accuracy(classifier_DT, X_testing, y_testing))
    print("Accuracy of NB", get_accuracy(classifier_NB, X_testing, y_testing))

    plt.show()
