import numpy as np

# pta stands for percentage by tick accuracy and is a metric that is used to evaluate the performance of the model.
# pta0 is the percentage of examples where the model's prediction is within 0 ticks of the true class, which is the
# same as the accuracy. pta1 is the percentage of examples where the model's prediction is within 1 tick of the true
# class (i.e. the model is off by 1 tick) and so on.


def pta(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ticks = np.abs(y_true - y_pred)
    pta = np.zeros(4)
    for i in range(4):
        pta[i] = np.sum(ticks <= i) / len(y_true)
    return pta
