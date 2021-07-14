import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    #prediction
    positives_predicted = np.count_nonzero(prediction)
    negatives_predicted = np.count_nonzero(prediction == False)
    count_all = prediction.shape[0]
    assert(positives_predicted + negatives_predicted == count_all)

    positives = np.count_nonzero(ground_truth) #relevant positives
    negatives = np.count_nonzero(ground_truth == False) #relevant negatives
    assert(positives + negatives == count_all)

    
    #print(prediction)
    #print(ground_truth)


    accurate_mask = (prediction == ground_truth)
    accurates = np.count_nonzero(accurate_mask)
    #print(accurate_mask)
    #print(accurates)

    true_positives_mask = accurate_mask == (accurate_mask <= prediction)
    true_positives = np.count_nonzero(true_positives_mask)
    #print(true_positives)
    #print(positives_predicted)


    precision = true_positives / positives_predicted
    recall = true_positives / positives
    accuracy = accurates / count_all
    f1 = 2 * precision * recall / (precision + recall)

    # implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # Implement computing accuracy
    count_all = prediction.shape[0]
    accurate_mask = (prediction == ground_truth)
    accurates = np.count_nonzero(accurate_mask)
    accuracy = accurates / count_all

    return accuracy
