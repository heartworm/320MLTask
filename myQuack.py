import csv
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sklearn as sk

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

    random.shuffle(X_pylist)
    random.shuffle(y_pylist)

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

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
#    raise NotImplementedError()


    classifier = DecisionTreeClassifier()
    classifier.fit(X_training, y_training)
    return classifier
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
#    raise NotImplementedError()
    
    k = 20 #number of neighbours to consider
    #'uniform' or 'distance' to prioritise neighbours votes based on proximity
    weighting = 'distance'
    
    classifier = KNeighborsClassifier(k, weights = weighting)
    classifier.fit(X_training, y_training)
    return classifier
    
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
#    raise NotImplementedError()

    classifier = SVC()
    classifier.fit(X_training, y_training)
    return classifier
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    TRAINING_SETASIDE = 0.75

    X, y = prepare_dataset("medical_records.data")

    num_points = X.shape[0]
    training_num = round(num_points * TRAINING_SETASIDE)

    X_training = X[0:training_num]
    X_testing = X[training_num:num_points]

    y_training = y[0:training_num]
    y_testing = y[training_num:num_points]

    classifier_NN = build_NN_classifier(X_training, y_training)
    classifier_SVM = build_SVM_classifier(X_training, y_training)
    classifier_DT = build_DT_classifier(X_training, y_training)
    classifier_NB = build_NB_classifier(X_training, y_training)
    pred_testing = classifier_NN.predict(X_testing)
    print("accuracy ", sk.metrics.accuracy_score(y_testing,pred_testing))
    


