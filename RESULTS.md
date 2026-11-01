# Results: Tawseel failed-delivery baseline


## Baseline table

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---|---|---|---|
| Dummy (prior) | 0.5000 | 0.2200 | 0.1717 | 0.22 |
| Rule: Vague or empty address| 0.5505 | 0.2419 | 0.3513 | 0.22 |
| My model | 0.6372 | 0.3635 | 0.1639 | 0.22 |

## Three sentences for Sherif

The model uses parcel information available at creation time to predict the risk of failed delivery.
It achieved a PR-AUC of 0.3635, outperforming the dummy baseline (0.2200) and the vague address rule (0.2419), with a PR-AUC lift of 0.1435 over the dummy,
however it should be under caution that it's a generated data and needs more validation and also like a 2 step verification after the model predicts a future failure attempt.
