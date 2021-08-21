from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from obp.policy import IPWLearner
from obp.types import BanditFeedback
from sklearn.linear_model import LogisticRegression

from pokemon_ml_design.actions import ACTIONS


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data: BanditFeedback) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: np.ndarray) -> List[int]:
        raise NotImplementedError


class IPWModel(BaseModel):
    def __init__(self) -> None:
        self._model = IPWLearner(
            n_actions=len(ACTIONS),
            base_classifier=LogisticRegression(C=100, random_state=12345, max_iter=1000)
        )

    def fit(self, data: BanditFeedback) -> None:
        self._model.fit(
            context=data["context"],
            action=data["action"],
            reward=data["reward"],
            pscore=data["pscore"],
        )

    def predict(self, context: np.ndarray) -> List[int]:
        return self._model.predict(context=context)
