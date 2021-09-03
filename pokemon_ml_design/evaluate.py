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
    # 意思決定モデルの性能評価を一気通貫で行うための`OffPolicyEvaluation`を定義する
    ope = OffPolicyEvaluation(
        bandit_feedback=validation_data, # バリデーションデータ
        ope_estimators=[IPS(estimator_name="IPS")] # 使用する推定量
    )

    # IPWLearner+ロジスティック回帰の性能をIPS推定量とDR推定量で評価
    ope.visualize_off_policy_estimates_of_multiple_policies(
        policy_name_list=list(action_choices.keys()),
        action_dist_list=list(action_choices.values()),
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

    fig = plt.figure(figsize = (8, 12))
    ax = fig.add_subplot(111)

    ax.axhline(ACTIONS[4].cost, ls = "-.", color = "magenta")  # マスターボールの価格にラインを引く

    for i, action in enumerate(ACTIONS):
        plt.scatter(capture_dificulties[i], rewards[i], label=action.label_en, c=action.color, marker=action.marker)

    plt.legend(fontsize='xx-large')
    plt.xlabel('Capture difficulty')
    plt.ylabel('Reward')
    plt.savefig('./resources/output/scatter.png')
    plt.clf()
