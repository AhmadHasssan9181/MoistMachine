import numpy as np

def compute_confusion_matrix(preds, labels, num_classes, ignore_index=255):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = (labels >= 0) & (labels < num_classes) & (labels != ignore_index)
    labels_i = labels[mask].astype(int)
    preds_i = preds[mask].astype(int)
    hist = np.bincount(num_classes * labels_i + preds_i,
                       minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes)

def compute_miou(hist):
    inter = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - inter
    iou = inter / (union + 1e-9)
    return np.nanmean(iou), iou