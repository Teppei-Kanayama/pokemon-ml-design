from typing import Optional

import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.utils import sigmoid
import pandas as pd

from pokemon_ml_design.actions import ACTIONS
from pokemon_ml_design.pokemon import PokemonZukan
from pokemon_ml_design.policy import rule_based_policy

# TODO: type hint
# TODO: shidoさんのやつを引用する


# 連続値をポケモンIDに変換する関数
# 現状のSyntheticBanditDatasetは整数値のcontextを生成できないので、暫定対応
def _get_pokemon_id(x: float) -> int:
    return (x * 10 ** 5).astype(int) % 151 + 1


def _reward_function(
    context: np.ndarray,  # 特徴量ベクトル  # context.shape = (n_rounds, dim_context)
    action_context: np.ndarray,  # 行動を表現するone-hotベクトル (n_actions, n_actions)次元の単位行列
    random_state: Optional[int] = None,
) -> np.ndarray:

    pokemon_ids = _get_pokemon_id(context.flatten())
    pokemon_zukan = PokemonZukan()

    capture_probabilities = []
    for pokemon_id in pokemon_ids:
        l = []  # TODO: 命名見直し
        for action in ACTIONS:
            ball_performance = action.performance
            pokemon_level = pokemon_zukan.get_capture_dificulty(pokemon_id)
            prob = sigmoid(ball_performance - pokemon_level)  # TODO: 確率計算する関数を変えてもよさそう
            l.append(prob)
        capture_probabilities.append(l)

    capture_probabilities = np.array(capture_probabilities)
    return capture_probabilities


def _behavior_policy(
    context: np.ndarray,  # 特徴量ベクトル  # context.shape = (n_rounds, dim_context)
    action_context: np.ndarray,  # 行動を表現するone-hotベクトル (n_actions, n_actions)次元の単位行列
    random_state: Optional[int] = None,
) -> np.ndarray:
    pokemon_ids = _get_pokemon_id(context.flatten())
    policy = np.array([rule_based_policy(pokemon_id) for pokemon_id in pokemon_ids])
    return policy


def _update_reward(data):
    # 捕獲したかどうかのrewardになっている。
    # 捕獲した場合は謝礼金をもらえて、捕獲しなかった場合は何ももらえない
    # ボールのコストを差し引く
    pokemon_ids = _get_pokemon_id(data['context'].flatten())
    pokemon_zukan = PokemonZukan()
    rewards = np.array([pokemon_zukan.get_reward(pokemon_id) for pokemon_id in pokemon_ids])
    costs = np.array([ACTIONS[action_id].cost for action_id in data['action'].flatten()])
    data['reward'] = rewards * data['reward'] - costs
    return data


def synthesize_data():
    # `SyntheticBanditDataset`を用いて人工データを生成する
    dataset = SyntheticBanditDataset(
        n_actions=5, # 人工データにおける行動の数  # 逃げる・モンスターボール・スーパーボール・ハイパーボール・マスターボール
        dim_context=1, # 人工データにおける特徴量の次元数 # 今回はポケモンIDを生成するだけで良いので1とする。
        reward_function=_reward_function, # 目的変数を生成する関数
        behavior_policy_function=_behavior_policy, # 過去の意思決定モデル\pi_bによる行動選択確率を生成する関数
        random_state=615,
    )

    # トレーニングデータとバリデーションデータを生成する
    print('train')  # TODO: printしない
    training_data = _update_reward(dataset.obtain_batch_bandit_feedback(n_rounds=1000))

    print('validation')
    validation_data = _update_reward(dataset.obtain_batch_bandit_feedback(n_rounds=1000))

    return training_data, validation_data
