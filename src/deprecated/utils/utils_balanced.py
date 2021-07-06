import torch
import sklearn.metrics as metrics


def metrics_(outputs,true_labels,class_to_search_on):
    precision = torch.Tensor(
        metrics.precision_score(
            true_labels.to("cpu"),
            outputs.to("cpu"),
            average=None,
            zero_division=0,
        )
    )
    recall = torch.Tensor(
        metrics.recall_score(
            true_labels.to("cpu"),
            outputs.to("cpu"),
            average=None,
            zero_division=0,
        )
    )
    
    report = metrics.classification_report(
        true_labels.to("cpu"),
        outputs.to("cpu"),
        zero_division=0,
        labels=class_to_search_on
    )
    
    return precision, recall,report
