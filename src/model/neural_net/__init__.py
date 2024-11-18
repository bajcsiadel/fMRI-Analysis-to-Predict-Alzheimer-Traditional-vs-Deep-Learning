import warnings

import numpy as np
import torch.optim
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.logger import TrainLogger


class SklearnWrapper(BaseEstimator, ClassifierMixin):

    count = {}

    def __init__(
        self,
        model: nn.Module,
        logger: TrainLogger,
        state_dict: dict = None,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer = torch.optim.Adam,
        lr: float = 0.001,
        criterion = torch.nn.CrossEntropyLoss,
        device: str | torch.device = "cpu",
        run_index: int = 0,
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
        self.experiment_name = f"E={self.epochs}-BS={self.batch_size}-optimizer={self.optimizer_type.__name__}-LR={self.lr}"

        self.__is_fitted = False
        self.run_index = run_index
        self.current_epoch = 0

        self.logger = logger
        self.tensorboard = SummaryWriter(str(self.logger.tensorboard_dir / self.experiment_name))
        self.__max_accuracy = 0.5

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
                expected = target[start_index:start_index + self.batch_size].type(torch.int64).to(self.device)
                y_batch = F.one_hot(expected, num_classes=self.model.n_classes).type(torch.FloatTensor).to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)

                _, predicted = torch.max(output.data, 1)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

                accuracy = (predicted == expected).sum() / len(expected)
                total_accuracy += (predicted == expected).sum()

                self.tensorboard.add_scalars("Stepwise/loss/train", {f"run-{self.run_index}": loss.item()}, self.current_epoch * steps + batch_number)
                self.tensorboard.add_scalars("Stepwise/accuracy/train", {f"run-{self.run_index}": accuracy}, self.current_epoch * steps + batch_number)

                total_loss += loss.item()

            self.tensorboard.add_scalars("Epochwise/loss/train", {f"run-{self.run_index}": total_loss / steps}, self.current_epoch)
            self.tensorboard.add_scalars("Epochwise/accuracy/train", {f"run-{self.run_index}": total_accuracy / len(X)}, self.current_epoch)
            self.tensorboard.add_scalars(f"Epochwise/loss/run-{self.run_index}", {"train": total_loss / steps}, self.current_epoch)
            self.tensorboard.add_scalars(f"Epochwise/accuracy/run-{self.run_index}", {"train": total_accuracy / len(X)}, self.current_epoch)

            validation_accuracy = self.score(X_valid, y_valid, test=True)
            # if validation_accuracy > self.__max_accuracy:
            #     self.__max_accuracy = validation_accuracy
            #     (self.logger.checkpoint_dir / self.experiment_name).mkdir(exist_ok=True)
            #     torch.save(
            #         {
            #             "model": self.model.state_dict(),
            #             "optimizer": self.optimizer.state_dict(),
            #             "epoch": self.current_epoch,
            #         },
            #         self.logger.checkpoint_dir
            #             / self.experiment_name
            #             / f"run-{self.run_index}-"
            #               f"epoch-{self.current_epoch}-"
            #               f"accu-{validation_accuracy:.4f}.pt"
            #     )

        self.__is_fitted = True
        return self

    # def __del__(self):
    #     self.tensorboard.close()

    def __sklearn_is_fitted__(self) -> bool:
        return self.__is_fitted

    def __sklearn_clone__(self) -> "SklearnWrapper":
        if self.experiment_name not in self.count:
            self.count[self.experiment_name] = 0
        self.count[self.experiment_name] += 1
        return SklearnWrapper(
            self.model,
            self.logger,
            self.state_dict,
            self.epochs,
            self.batch_size,
            self.optimizer_type,
            self.lr,
            self.criterion_type,
            self.device,
            self.count[self.experiment_name],
        )

    def _predict(self, X: torch.Tensor, y: torch.Tensor = None) -> (torch.Tensor, float):
        """
        Predict the target using the model.
        :param X: input data
        :param y: target
        :return: output logits and model loss
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)

            match y:
                case np.ndarray():
                    y = torch.from_numpy(y)
                case torch.Tensor() | None:
                    # do nothing
                    pass
                case _:
                    warnings.warn(f"y is not a numpy array or torch tensor {type(y)}")

            output = self.model(X)

            if y is not None:
                y = F.one_hot(y.type(torch.int64), num_classes=self.model.n_classes).type(torch.FloatTensor).to(self.device)
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
        self.tensorboard.add_scalars(f"Epochwise/accuracy/{set_name}", {f"run-{self.run_index}": accuracy}, self.current_epoch)
        self.tensorboard.add_scalars(f"Epochwise/loss/{set_name}", {f"run-{self.run_index}": loss}, self.current_epoch)
        self.tensorboard.add_scalars(f"Epochwise/loss/run-{self.run_index}", {set_name: loss}, self.current_epoch)
        self.tensorboard.add_scalars(f"Epochwise/accuracy/run-{self.run_index}", {set_name: accuracy}, self.current_epoch)
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

    def set_params(self, **params) -> "SklearnWrapper":
        """
        Set the parameters of the model.
        """
        super().set_params(**params)

        self.optimizer = self.optimizer_type([
            {"params": self.model.parameters(), "lr": self.lr}
        ])
        self.criterion = self.criterion_type()
        return self