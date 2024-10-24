from pathlib import Path

import numpy as np
import torch.optim
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class SklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: nn.Module,
        state_dict: dict = None,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer = torch.optim.Adam,
        lr: float = 0.001,
        criterion = torch.nn.CrossEntropyLoss,
        device: str | torch.device = "cpu",
        tensorboard_dir: Path = Path("tensorboard"),
        run_index: int = None,
    ):
        self.model = model
        if state_dict is not None:
            self.state_dict = state_dict
            self.model.load_state_dict(state_dict)
        else:
            self.state_dict = model.state_dict()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_type = optimizer
        self.optimizer = optimizer([
            {"params": self.model.parameters(), "lr": lr}
        ])
        self.criterion_type = criterion
        self.criterion = criterion()
        self.device = device

        self.__is_fitted = False
        self.run_index = 1 if run_index is None else run_index
        self.tensorboard_dir = tensorboard_dir
        self.tensorboard = SummaryWriter(str(
            self.tensorboard_dir / f"run_{self.run_index}-E_{self.epochs}-"
                                   f"BC_{self.batch_size}-"
                                   f"optim_{self.optimizer_type.__name__}-LR_{self.lr}-"
                                   f"criterion_{self.criterion_type.__name__}"
        ))
        self.current_epoch = 0

    def fit(self, X, y, target, X_valid=None, y_valid=None) -> "SklearnWrapper":
        """
        Fit the model to the data.
        :param X: input data
        :param y: target in a numpy array used for the definition of cross-validation
        :param target: target in a torch tensor
        :param X_valid: validation input data
        :param y_valid: validation target
        :return: self
        """
        steps = len(X) // self.batch_size
        for self.current_epoch in np.arange(self.epochs) + 1:
            self.model.train()
            total_loss = 0
            total_accuracy = 0
            for batch_number, start_index in enumerate(range(0, len(X), self.batch_size)):
                X_batch = X[start_index:start_index + self.batch_size].to(self.device)
                y_batch = target[start_index:start_index + self.batch_size].type(torch.LongTensor).to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)

                _, predicted = torch.max(output.data, 1)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

                accuracy = (predicted == y_batch).sum() / len(y_batch)
                total_accuracy += (predicted == y_batch).sum()

                self.tensorboard.add_scalar("Loss/train/step", loss.item(), self.current_epoch * steps + batch_number)
                self.tensorboard.add_scalar("Accuracy/train/step", accuracy, self.current_epoch * steps + batch_number)

                total_loss += loss.item()

            self.tensorboard.add_scalar("Loss/train/epoch", total_loss / steps, self.current_epoch)
            self.tensorboard.add_scalar("Accuracy/train/epoch", total_accuracy / len(X), self.current_epoch)

            self.score(X_valid, y_valid, test=True)

        self.__is_fitted = True
        return self

    def __del__(self):
        self.tensorboard.close()

    def __sklearn_is_fitted__(self):
        return self.__is_fitted

    def _predict(self, X: torch.Tensor, y: torch.Tensor = None) -> (torch.Tensor, float):
        """
        Predict the target using the model.
        :param X: input data
        :param y: target
        :return: output logits and model loss
        """
        with torch.no_grad():
            self.model.eval()
            X = X.to(self.device)

            match y:
                case np.ndarray():
                    y = torch.from_numpy(y).type(torch.LongTensor).to(self.device)
                case torch.Tensor():
                    y = y.type(torch.LongTensor).to(self.device)

            output = self.model(X)

            if y is not None:
                loss = self.criterion(output, y).item()
            else:
                loss = None

            return output, loss

    def predict(self, X: torch.Tensor, y: torch.Tensor = None) -> (np.ndarray, float):
        """
        Predict the target using the model.
        :param X: input data
        :param y: target
        :return: predicted target and model loss
        """
        output, loss = self._predict(X, y)
        return output.argmax(dim=1).cpu().detach().numpy(), loss

    def predict_proba(self, X: torch.Tensor, y: torch.Tensor = None) -> (np.ndarray, float):
        """
        Predict the probabilities of each class.
        :param X: input data
        :param y: target
        :return: predicted probabilities in shape (n_samples, n_classes)
        """
        output, loss = self._predict(X, y)
        # convert logits to probabilities
        return F.softmax(output, dim=1).cpu().detach().numpy(), loss

    def score(self, X: torch.Tensor, y: torch.Tensor, sample_weight=None, test: bool = False) -> float:
        """
        Compute the accuracy of the model.
        :param X: input data
        :param y: target
        :param sample_weight: sample weights
        :param test: whether to use the test set or validation set
        :return: accuracy of the model
        """
        set_name = "test" if test else "validation"
        output, loss = self.predict(X, y)
        accuracy = accuracy_score(y, output, sample_weight=sample_weight)
        self.tensorboard.add_scalar(f"Accuracy/{set_name}", accuracy, self.current_epoch)
        self.tensorboard.add_scalar(f"Loss/{set_name}", loss, self.current_epoch)
        return accuracy

    def get_params(self, deep: bool = False) -> dict:
        """
        Get the parameters of the model used in the constructor.
        :param deep: whether to get the parameters of the model
        """
        params = super().get_params(deep)
        params["optimizer"] = self.optimizer_type
        params["criterion"] = self.criterion_type
        return params

    def set_params(self, **params):
        """
        Set the parameters of the model.
        """
        super().set_params(**params)

        self.optimizer = self.optimizer_type([
            {"params": self.model.parameters(), "lr": self.lr}
        ])
        self.criterion = self.criterion_type()
        self.tensorboard = SummaryWriter(str(self.tensorboard_dir))
        self.run_index += 1
        return self