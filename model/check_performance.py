import time
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EvaluationMetrics:
    def __init__(self):
        self.learning_time_list = list()

        self.train_accuracy_list = list()
        self.train_precision_list = list()
        self.train_recall_list = list()
        self.train_f1core_list = list()

        self.test_accuracy_list = list()
        self.test_precision_list = list()
        self.test_recall_list = list()
        self.test_f1core_list = list()

    def calculate(self, learning_time, y_train_true, y_train_pred, y_test_true, y_test_pred):
        self.learning_time_list.append(learning_time)

        self.train_accuracy_list.append(accuracy_score(y_train_true, y_train_pred))
        self.train_precision_list.append(precision_score(y_train_true, y_train_pred, average='macro'))
        self.train_recall_list.append(recall_score(y_train_true, y_train_pred, average='macro'))
        self.train_f1core_list.append(f1_score(y_train_true, y_train_pred, average='macro'))

        self.test_accuracy_list.append(accuracy_score(y_test_true, y_test_pred))
        self.test_precision_list.append(precision_score(y_test_true, y_test_pred, average='macro'))
        self.test_recall_list.append(recall_score(y_test_true, y_test_pred, average='macro'))
        self.test_f1core_list.append(f1_score(y_test_true, y_test_pred, average='macro'))

    def get_result(self):
        np.set_printoptions(suppress=True)

        msg = "-------------- Evaluation Result --------------"
        msg += f"\n Number of evaluation:       {len(self.learning_time_list)}"
        msg += f"\n Learning time:              {np.mean(self.learning_time_list):.03f}"
        msg += f"\n --"
        msg += f"\n Train accuracy:             {np.mean(self.train_accuracy_list):.03f}"
        msg += f"\n Train precision:            {np.mean(self.train_precision_list):.03f}"
        msg += f"\n Train recall:               {np.mean(self.train_recall_list):.03f}"
        msg += f"\n Train f1-score:             {np.mean(self.train_f1core_list):.03f}"
        msg += f"\n --"
        msg += f"\n Test accuracy:              {np.mean(self.test_accuracy_list):.03f}"
        msg += f"\n Test precision:             {np.mean(self.test_precision_list):.03f}"
        msg += f"\n Test recall:                {np.mean(self.test_recall_list):.03f}"
        msg += f"\n Test f1-score:              {np.mean(self.test_f1core_list):.03f}"
        msg += "\n\n"

        return msg


class PerformanceChecker:
    def __init__(self, classifier=None, cv=5):
        if classifier is None:
            raise Exception("[ERROR] Invalid classifier.")

        self.classifier = classifier
        self.cv = cv

    def start(self, X=None, y=None):
        if (X is None) or (y is None):
            raise Exception("[ERROR] Invalid parameters.")

        evaluation_metrics = EvaluationMetrics()

        kfold = StratifiedKFold(n_splits=self.cv, shuffle=True)
        for train_indexes, test_indexes in kfold.split(X, y):
            X_train_data = X[train_indexes]
            X_test_data = X[test_indexes]
            y_train_data = y[train_indexes]
            y_test_data = y[test_indexes]

            classifier = clone(self.classifier)

            start_time = time.time()
            classifier.fit(X_train_data, y_train_data)
            elapsed = time.time() - start_time

            y_train_pred = classifier.predict(X_train_data)
            y_test_pred = classifier.predict(X_test_data)

            evaluation_metrics.calculate(elapsed, y_train_data, y_train_pred, y_test_data, y_test_pred)

        print(evaluation_metrics.get_result())

