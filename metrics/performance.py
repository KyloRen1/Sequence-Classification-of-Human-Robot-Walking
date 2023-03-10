from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, 
    recall_score, accuracy_score
)

def compute_model_metrics(labels_ds, preds, CLASS_MAP, average='weighted'):
    ''' Compute metrics:
            - Accuracy score
            - F1-score
            - Precision
            - Recall
            - Confusion matrix
    '''
    cmat = confusion_matrix(labels_ds, preds, labels=range(len(CLASS_MAP)))
    # normalizing confusion matrix
    cmat = (cmat.T / cmat.sum(axis=1)).T
    f1 = f1_score(
        labels_ds, preds, labels=range(
            len(CLASS_MAP)), average=average)
    precision = precision_score(
        labels_ds, preds, labels=range(
            len(CLASS_MAP)), average=average)
    recall = recall_score(
        labels_ds, preds, labels=range(
            len(CLASS_MAP)), average=average)
    accuracy = accuracy_score(labels_ds, preds)
    return accuracy, f1, precision, recall, cmat