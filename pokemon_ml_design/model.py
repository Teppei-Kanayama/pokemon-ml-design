from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from obp.policy import IPWLearner, NNPolicyLearner
from obp.types import BanditFeedback
from obp.ope import DirectMethod, InverseProbabilityWeighting
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from pokemon_ml_design.actions import ACTIONS
from pokemon_ml_design.policy import rule_based_policy, deterministic_rule_based_policy


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data: BanditFeedback) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: np.ndarray) -> List[int]:
        raise NotImplementedError


class IPWModel(BaseModel):
    def __init__(self) -> None:
        self._model = NNPolicyLearner(
            n_actions=len(ACTIONS),
            dim_context=2,
            off_policy_objective=InverseProbabilityWeighting().estimate_policy_value_tensor,  # TODO: これはなに？
            random_state=615
        )
        self._scaler = StandardScaler()

    def fit(self, data: BanditFeedback) -> None:
        context = data["context"][:, 1:]
        self._scaler.fit(context)
        scaled_context = self._scaler.transform(context)

        self._model.fit(
            context=scaled_context,
            action=data["action"],
            # reward=data["binary_reward"],
            reward=data["reward"],
            pscore=data["pscore"],
        )

    def predict(self, context: np.ndarray) -> np.ndarray:
        scaled_context = self._scaler.transform(context[:, 1:])
        return self._model.predict(context=scaled_context)


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


class DeterministicRuleBasedModel(BaseModel):
    def __init__(self) -> None:
        pass

    def fit(self, data: BanditFeedback) -> None:
        # 学習フェーズはない
        pass

    def predict(self, context: np.ndarray) -> np.ndarray:
        pokemon_ids = context[:, 0]
        predictions = []
        for pokemon_id in pokemon_ids:
            prediction = deterministic_rule_based_policy(pokemon_id)
            predictions.append(np.array(prediction))
        return  np.array(predictions)[:, :, np.newaxis]
