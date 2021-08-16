import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    count_all = prediction.shape[0]
    accurate_mask = (prediction == ground_truth)
    accurates = np.count_nonzero(accurate_mask)
    accuracy = accurates / count_all

    return accuracy
