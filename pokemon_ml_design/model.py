from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from obp.policy import IPWLearner
from obp.types import BanditFeedback
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from pokemon_ml_design.actions import ACTIONS
from pokemon_ml_design.policy import rule_based_policy


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
            base_classifier=LogisticRegression(C=100, random_state=615, max_iter=1000)
        )
        self._scaler = StandardScaler()

    def fit(self, data: BanditFeedback) -> None:
        self._scaler.fit(data["context"])
        scaled_context = self._scaler.transform(data["context"])

        self._model.fit(
            context=scaled_context[:, 1:],  # TODO: リファクタ
            action=data["action"],
            reward=data["reward"],
            pscore=data["pscore"],
        )

    def predict(self, context: np.ndarray) -> np.ndarray:
        scaled_context = self._scaler.transform(context)
        return self._model.predict(context=scaled_context[:, 1:])


class RuleBasedModel(BaseModel):
    def __init__(self) -> None:
        pass

    def fit(self, data: BanditFeedback) -> None:
        # 学習フェーズはない
        pass

    def predict(self, context: np.ndarray) -> np.ndarray:
        pokemon_ids = context[:, 0]
        predictions = []
        for pokemon_id in pokemon_ids:
            probabilities = rule_based_policy(pokemon_id)
            prediction = np.random.multinomial(n=1, pvals=probabilities, size=1)[0]
            predictions.append(prediction)
        return  np.array(predictions)[:, :, np.newaxis]
