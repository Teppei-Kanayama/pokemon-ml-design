from typing import List

from obp.policy import IPWLearner
from obp.types import BanditFeedback
from sklearn.linear_model import LogisticRegression

from pokemon_ml_design.actions import ACTIONS

# TODO: 抽象クラスを作る

class IPWModel:
    def __init__(self) -> None:
        self._model = IPWLearner(
            n_actions=len(ACTIONS),
            base_classifier=LogisticRegression(C=100, random_state=12345, max_iter=1000)
        )

    def fit(self, data: BanditFeedback) -> None:
        self._model.fit(
            context=data["context"], # 特徴量
            action=data["action"], # 過去の意思決定モデル\pi_bによる行動選択
            reward=data["reward"], # 観測される目的変数
            pscore=data["pscore"], # 過去の意思決定モデル\pi_bによる行動選択確率(傾向スコア)
        )

    def predict(self, data: BanditFeedback) -> List[int]:  # TODO: contextだけ渡す
        return self._model.predict(context=data["context"])
