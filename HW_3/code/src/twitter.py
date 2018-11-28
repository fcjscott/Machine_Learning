"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """


    return np.genfromtxt(fname)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        value = 0
        for line in fid.readlines():
            words = extract_words(line)
            for word in words:
                if word in word_list.keys():
                    pass
                else:
                    word_list[word] = value
                    value += 1
        pass
        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix

        i = 0
        for line in fid.readlines():
            words = extract_words(line)
            for word in words:
                ids = word_list[word]
                feature_matrix[i][ids] = 1
            i += 1

        pass
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc'       
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label == 0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_label)
    else:
        raise ValueError("metric not set correctly")

    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance
    score = []
    for train_id, cv_id in kf.split(X, y):
        X_train, X_cv = X[train_id], X[cv_id]
        y_train, y_cv = y[train_id], y[cv_id]
        clf.fit(X_train, y_train)
        y_pred = clf.decision_function(X_cv)
        score.append(performance(y_cv, y_pred, metric=metric))

    return np.average(score)
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2: select optimal hyperparameter using cross-validation
    score = []
    for c in C_range:
        clf = SVC(kernel='linear', C=c)
        score.append(cv_performance(clf, X, y, kf, metric=metric))

    max_id = score.index(max(score))

    return C_range[max_id]
    ### ========== TODO : END ========== ###



def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 3: return performance on test data by first computing predictions and then calling performance
    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric=metric)
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    print(dictionary)
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    print(X.shape)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1-score", "auroc"]
    
    ### ========== TODO : START ========== ###
    # part 1: split data into training (training + cross-validation) and testing set
    X_training = X[:560][:]
    X_testing = X[560:][:]
    y_training = y[:560]
    y_testing = y[560:]
    
    # part 2: create stratified folds (5-fold CV)
    kf = StratifiedKFold(n_splits=5)
    
    # part 2: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    C = select_param_linear(X_training, y_training, kf=kf, metric="accuracy")
        
    # part 3: train linear-kernel SVMs with selected hyperparameters

    svm_acc = SVC(kernel='linear', C=10)
    svm_acc.fit(X_training, y_training)
    svm_f1 = SVC(kernel='linear', C=10)
    svm_f1.fit(X_training, y_training)
    svm_auroc = SVC(kernel='linear', C=1)
    svm_auroc.fit(X_training, y_training)
    # part 3: report performance on test data

    score_acc = performance_test(clf=svm_acc, X=X_testing, y=y_testing, metric='accuracy')
    score_f1 = performance_test(clf=svm_f1, X=X_testing, y=y_testing, metric='f1-score')
    score_auroc = performance_test(clf=svm_auroc, X=X_testing, y=y_testing, metric='auroc')
    print("accuracy score is: %.3f" % score_acc)
    print("f1 score is: %.3f" % score_f1)
    print("auroc score is: %.3f" % score_auroc)
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
