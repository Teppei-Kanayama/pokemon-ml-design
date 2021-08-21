from typing import Optional

import numpy as np
from obp.dataset import (
    SyntheticBanditDataset,
)
from obp.utils import sigmoid
import pandas as pd
from pathlib import Path


# TODO: type hint

# ポケモンの捕獲容易度と謝礼金を教えてくれるポケモン図鑑クラス
class PokemonZukan:
    def __init__(self):
        self._data = pd.read_csv('resources/input/pokemon.csv', index_col='id')

    def get_capture_dificulty(self, pokemon_id):
        return self._data.loc[pokemon_id]['capture_dificulty']

    def get_reward(self, pokemon_id):
        return self._data.loc[pokemon_id]['reward']

    def get_name(self, pokemon_id):
        return self._data.loc[pokemon_id]['name']


# TODO: classとして持つには大げさか？
class Ball:
    def __init__(self, ball_type: str) -> None:
        self._ball_coefficient_mapper = dict(
            monster=0,
            super=100,
            hyper=300,
            master=None,
        )
        self.ball_type = ball_type

    def get_coefficient(self) -> int:
        return self._ball_coefficient_mapper[self.ball_type]

# 100倍した方が迫力が出るかもしれない
ball_price = {0: 0, 1: 50, 2: 100, 3: 200, 4: 500}


def get_pokemon_id(v):
    return (v * 10 ** 5).astype(int) % 151 + 1


def my_reward_function(
    context: np.ndarray,  # 特徴量ベクトル  # context.shape = (n_rounds, dim_context)
    action_context: np.ndarray,  # 行動を表現するone-hotベクトル (n_actions, n_actions)次元の単位行列
    random_state: Optional[int] = None,
) -> np.ndarray:

    pokemon_ids = get_pokemon_id(context.flatten())
    pokemon_zukan = PokemonZukan()

    capture_probabilities = []
    for pokemon_id in pokemon_ids:
        l = []
        for action in ['run_away', Ball('monster'), Ball('super'), Ball('hyper'), Ball('master')]:  # TODO: リファクタ
            if action == 'run_away':
                l.append(0)
            elif action.ball_type == 'master':
                l.append(1)
            else:
                ball_coefficient = action.get_coefficient()
                pokemon_level = pokemon_zukan.get_capture_dificulty(pokemon_id)
                prob = sigmoid(ball_coefficient - pokemon_level)  # TODO: 確率計算する関数を変えてもよさそう
                l.append(prob)
            capture_probabilities.append(l)

    capture_probabilities = np.array(capture_probabilities)
    return capture_probabilities


def rule_based_policy(pokemon_id):
    # TODO: 戻り値をhashにした方が読みやすそう
    pokemon_zukan = PokemonZukan()

    capture_dificulty = pokemon_zukan.get_capture_dificulty(pokemon_id)
    reward = pokemon_zukan.get_reward(pokemon_id)
    name = pokemon_zukan.get_name(pokemon_id)

    # TODO: もう少し練る
    if name == 'ミュウツー':
        return 0, 0, 0, 0.2, 0.8

    if capture_dificulty >= 200 and reward >= 300:
        return 0, 0.1, 0.2, 0.7, 0

    if capture_dificulty < 100 and reward < 200:
        return 0.5, 0.4, 0.1, 0, 0

    return 0.1, 0.3, 0.6, 0, 0


def my_behavior_policy(
    context: np.ndarray,  # 特徴量ベクトル  # context.shape = (n_rounds, dim_context)
    action_context: np.ndarray,  # 行動を表現するone-hotベクトル (n_actions, n_actions)次元の単位行列
    random_state: Optional[int] = None,
) -> np.ndarray:
    pokemon_ids = get_pokemon_id(context.flatten())
    policy = np.array([rule_based_policy(pokemon_id) for pokemon_id in pokemon_ids])
    return policy

def update_reward(data):
    # 捕獲したかどうかのrewardになっている。
    # 捕獲した場合は謝礼金をもらえて、捕獲しなかった場合は何ももらえない
    # ボールのコストを差し引く
    pokemon_ids = get_pokemon_id(data['context'].flatten())
    pokemon_zukan = PokemonZukan()
    rewards = np.array([pokemon_zukan.get_reward(pokemon_id) for pokemon_id in pokemon_ids])
    costs = np.array([ball_price[action] for action in data['action'].flatten()])
    data['reward'] = rewards * data['reward'] - costs
    return data

# `SyntheticBanditDataset`を用いて人工データを生成する
dataset = SyntheticBanditDataset(
    n_actions=5, # 人工データにおける行動の数  # 逃げる・モンスターボール・スーパーボール・ハイパーボール・マスターボール
    dim_context=1, # 人工データにおける特徴量の次元数 # 今回はポケモンIDを生成するだけで良いので1とする。
    reward_function=my_reward_function, # 目的変数を生成する関数
    behavior_policy_function=my_behavior_policy, # 過去の意思決定モデル\pi_bによる行動選択確率を生成する関数
    random_state=615,
)

def synthesize_data():
    # トレーニングデータとバリデーションデータを生成する
    print('train')  # TODO: printしない
    training_data = update_reward(dataset.obtain_batch_bandit_feedback(n_rounds=1000))

    print('validation')
    validation_data = update_reward(dataset.obtain_batch_bandit_feedback(n_rounds=1000))

    return training_data, validation_data
