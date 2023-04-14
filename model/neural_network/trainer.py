import torch
import numpy as np

from tqdm.auto import tqdm
from sklearn.metrics import f1_score


class Classifier:
    def __init__(self, model, optimizer, criterion, threshold, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.threshold = threshold
        self.device = device

        self.model.to(device)
        self.criterion.to(device)

        self.best_model = None

    def train(self, train_data_loader, validation_data_loader, epoch):
        best_score = 0
        best_model = None

        for epoch in range(1, epoch + 1):
            self.model.train()

            loss_list = list()
            for features, targets in tqdm(iter(train_data_loader)):
                features = features.float().to(self.device)
                targets = targets.float().to(self.device)

                self.optimizer.zero_grad()

                pred = self.model(features)
                loss = self.criterion(pred, targets.reshape(-1, 1))

                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())

            validation_loss, validation_score = self.validate(validation_data_loader)
            print(
                f"Epoch [{epoch}]   Train Loss: [{np.mean(loss_list):.05f}]   Validation Loss: [{validation_loss:.05f}]   Validation Score: [{validation_score:.05f}]")

            if self.scheduler is not None:
                self.scheduler.step(validation_score)

            if best_score < validation_score:
                best_score = validation_score
                best_model = self.model

        self.best_model = best_model

    # 모델 검증
    def validate(self, data_loader):
        self.model.eval()

        loss_list = list()
        y_true_list = list()
        y_pred_list = list()
        with torch.no_grad():
            for features, targets in tqdm(iter(data_loader)):
                y_true_list += targets.tolist()

                features = features.float().to(self.device)
                targets = targets.float().to(self.device)

                pred = self.model(features)
                loss = self.criterion(pred, targets.reshape(-1, 1))

                loss_list.append(loss.item())

                pred = pred.squeeze(1).to("cpu")
                y_pred_list += pred.tolist()

        y_pred_list = np.where(np.array(y_pred_list) > self.threshold, 1, 0)
        score = f1_score(y_true=y_true_list, y_pred=y_pred_list, average="macro")

        return np.mean(loss_list)

    # 추론
    def predict(self, data_loader):
        self.model.eval()

        y_pred_list = list()
        with torch.no_grad():
            for features in tqdm(iter(data_loader)):
                features = features.float().to(self.device)

                pred = self.best_model(features)
                pred = pred.squeeze(1).to("cpu")

                y_pred_list += pred.tolist()

        y_pred_list = np.where(np.array(y_pred_list) > self.threshold, 1, 0)

        return y_pred_list
