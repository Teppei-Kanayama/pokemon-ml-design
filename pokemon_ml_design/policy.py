from typing import Tuple

from pokemon_ml_design.pokemon import PokemonZukan


def rule_based_policy(pokemon_id: int) -> Tuple:
    pokemon_zukan = PokemonZukan()

    capture_dificulty = pokemon_zukan.get_capture_dificulty(pokemon_id)
    reward = pokemon_zukan.get_reward(pokemon_id)
    name = pokemon_zukan.get_name(pokemon_id)

    # if name == 'ミュウツー':
    #     return 0, 0, 0, 0.2, 0.8
    #
    # if capture_dificulty >= 200 and reward >= 300:
    #     return 0, 0.1, 0.2, 0.7, 0
    #
    # if capture_dificulty < 100 and reward < 200:
    #     return 0.5, 0.4, 0.1, 0, 0
    #
    # return 0.1, 0.3, 0.6, 0, 0

    return 0.2, 0.2, 0.2, 0.2, 0.2
