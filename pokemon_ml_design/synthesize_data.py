from typing import Optional, Tuple

import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.utils import sigmoid
from obp.types import BanditFeedback
import pandas as pd

from pokemon_ml_design.actions import ACTIONS
from pokemon_ml_design.pokemon import PokemonZukan
from pokemon_ml_design.policy import rule_based_policy


# 連続値をポケモンIDに変換する関数
# TODO: provate関数にして良さそう
def get_pokemon_id(x: float) -> int:
    return (x * 10 ** 5).astype(int) % 151 + 1


def _reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:

    pokemon_ids = get_pokemon_id(context.flatten())
    pokemon_zukan = PokemonZukan()

    capture_probabilities = []
    for pokemon_id in pokemon_ids:
        l = []
        for action in ACTIONS:
            ball_performance = action.performance
            pokemon_capture_dificulty = pokemon_zukan.get_capture_dificulty(pokemon_id)
            capture_probability = sigmoid(ball_performance - pokemon_capture_dificulty)
            l.append(capture_probability)
        capture_probabilities.append(l)
    capture_probabilities = np.array(capture_probabilities)
    return capture_probabilities


def _behavior_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    pokemon_ids = get_pokemon_id(context.flatten())
    policy = np.array([rule_based_policy(pokemon_id) for pokemon_id in pokemon_ids])
    return policy


# 現状のSyntheticBanditDatasetは整数値のcontextを生成できないので、暫定的に後処理でcontextを追加する
def _update_context(data: BanditFeedback) -> BanditFeedback:
    pokemon_ids = get_pokemon_id(data['context'].flatten())
    pokemon_zukan = PokemonZukan()
    rewards = np.array([pokemon_zukan.get_reward(pokemon_id) for pokemon_id in pokemon_ids])
    capture_dificulties = np.array([pokemon_zukan.get_capture_dificulty(pokemon_id) for pokemon_id in pokemon_ids])

    new_context = np.concatenate([pokemon_ids[:, np.newaxis], capture_dificulties[:, np.newaxis], rewards[:, np.newaxis]], axis=1)
    data['context'] = new_context
    return data


# 現状、捕獲したかどうかのrewardになっているので、
# 「捕獲した場合は謝礼金をもらえて、捕獲しなかった場合は何ももらえない」「ボールのコストを差し引く」を考慮したrewardにする
def _update_reward(data: BanditFeedback) -> BanditFeedback:
    rewards = data['context'][:, 2]
    costs = np.array([ACTIONS[action_id].cost for action_id in data['action'].flatten()])
    data['binary_reward'] = data['reward']
    data['reward'] = rewards * data['reward'] - costs
    return data


def _update_expected_reward(data: BanditFeedback) -> BanditFeedback:
    rewards = data['context'][:, 2]
    costs = np.array([action.cost for action in ACTIONS])
    data['expected_reward'] = rewards[:, np.newaxis] * data['expected_reward'] - costs[np.newaxis, :]
    return data


def _post_process(data: BanditFeedback) -> BanditFeedback:
    data = _update_context(data)
    data = _update_reward(data)
    data = _update_expected_reward(data)
    return data


def synthesize_data() -> Tuple[BanditFeedback, BanditFeedback]:
    dataset = SyntheticBanditDataset(
        n_actions=len(ACTIONS),
        dim_context=1,  # pokemon_idの元になるfloat値を生成する
        reward_function=_reward_function,
        behavior_policy_function=_behavior_policy,
        random_state=615,
    )
    training_data = _post_process(dataset.obtain_batch_bandit_feedback(n_rounds=50000))
    validation_data = _post_process(dataset.obtain_batch_bandit_feedback(n_rounds=50000))
    test_data = _post_process(dataset.obtain_batch_bandit_feedback(n_rounds=1000))
    return training_data, validation_data, test_data
