from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from obp.policy import IPWLearner
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DoublyRobust as DR
)
from obp.types import BanditFeedback

from pathlib import Path
from sklearn.linear_model import LogisticRegression

from pokemon_ml_design.actions import ACTIONS


def evaluate(validation_data: BanditFeedback, action_choices: Dict[str, List]) -> None:
    # TODO: 意味を理解する
    # TODO: 必要に応じて分離する

    # DR推定量に必要な目的変数予測モデルを得る
    # opeモジュールに実装されている`RegressionModel`に好みの機械学習手法を与えば良い
    regression_model = RegressionModel(
        n_actions=len(ACTIONS),
        base_model=LogisticRegression(C=100, random_state=12345, max_iter=1000), # ロジスティック回帰を使用
    )

    # `fit_predict`メソッドにより、バリデーションデータにおける期待報酬を推定
    estimated_rewards_by_reg_model = regression_model.fit_predict(
        context=validation_data["context"], # 特徴量
        action=validation_data["action"], # 過去の意思決定モデル\pi_bによる行動選択
        reward=validation_data["reward"], # 観測される目的変数
        random_state=12345,
    )

    # 意思決定モデルの性能評価を一気通貫で行うための`OffPolicyEvaluation`を定義する
    ope = OffPolicyEvaluation(
        bandit_feedback=validation_data, # バリデーションデータ
        ope_estimators=[IPS(estimator_name="IPS"), DR()] # 使用する推定量
    )

    # IPWLearner+ロジスティック回帰の性能をIPS推定量とDR推定量で評価
    ope.visualize_off_policy_estimates_of_multiple_policies(
        policy_name_list=list(action_choices.keys()),
        action_dist_list=list(action_choices.values()),
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model, # DR推定量に必要な期待報酬推定値
        random_state=12345,
        fig_dir=Path('./resources/output/'),
    )
    plt.clf()

    capture_dificulties = defaultdict(list)
    rewards = defaultdict(list)
    for action, context in zip(action_choices['IPW'], validation_data['context']):
        capture_dificulty = context[1]
        reward = context[2]
        action_id = np.argmax(action)

        capture_dificulties[action_id].append(capture_dificulty)
        rewards[action_id].append(reward)

    fig = plt.figure(figsize = (8, 20))
    ax = fig.add_subplot(111)
    ax.axhline(ACTIONS[1].cost, ls = "-.", color = "magenta")
    ax.axhline(ACTIONS[2].cost, ls = "-.", color = "magenta")
    ax.axhline(ACTIONS[3].cost, ls = "-.", color = "magenta")
    ax.axhline(ACTIONS[4].cost, ls = "-.", color = "magenta")

    for i, action in enumerate(ACTIONS):
        print(i, len(capture_dificulties[i]))
        plt.scatter(capture_dificulties[i], rewards[i], label=i)  # TODO:　action.labelを使う
    plt.legend()
    plt.savefig('./resources/output/scatter.png')
    plt.clf()
