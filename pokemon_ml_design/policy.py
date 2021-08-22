from typing import Tuple

from pokemon_ml_design.pokemon import PokemonZukan


def rule_based_policy(pokemon_id: int) -> Tuple:
    pokemon_zukan = PokemonZukan()

    capture_dificulty = pokemon_zukan.get_capture_dificulty(pokemon_id)
    reward = pokemon_zukan.get_reward(pokemon_id)
    name = pokemon_zukan.get_name(pokemon_id)

    if reward >= 10000:
        return 0.1, 0.1, 0.1, 0.2, 0.5

    if 2000 <= reward and reward < 10000:
        return 0.05, 0.2, 0.3, 0.4, 0.05

    if 500 <= reward and reward < 2000:
        return 0.1, 0.3, 0.5, 0.05, 0.05

    if 100 <= reward and reward < 500:
        return 0.4, 0.4, 0.1, 0.05, 0.05

    return 0.8, 0.05, 0.05, 0.05, 0.05
