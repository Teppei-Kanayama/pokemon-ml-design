from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_behavior_policy
)
from obp.policy import IPWLearner, Random
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DoublyRobust as DR
)
from obp.utils import softmax, sigmoid


# TODO: 抽象クラスを作る

class IPWModel:
    def __init__(self, n_actions: int) -> None:
        self._model = IPWLearner(
            n_actions=n_actions,
            base_classifier=LogisticRegression(C=100, random_state=12345, max_iter=1000)
        )

    def fit(self, data):
        self._model.fit(
            context=data["context"], # 特徴量
            action=data["action"], # 過去の意思決定モデル\pi_bによる行動選択
            reward=data["reward"], # 観測される目的変数
            pscore=data["pscore"], # 過去の意思決定モデル\pi_bによる行動選択確率(傾向スコア)
        )

    def predict(self, data):  # TODO: contextだけ渡す
        return self._model.predict(context=data["context"])
