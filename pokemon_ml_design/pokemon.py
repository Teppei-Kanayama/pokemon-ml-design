import pandas as pd


# ポケモンの捕獲容易度と謝礼金を教えてくれるポケモン図鑑クラス
class PokemonZukan:
    def __init__(self):
        self._data = pd.read_csv('resources/input/pokemon.csv', index_col='id')

    def get_capture_dificulty(self, pokemon_id: int) -> int:
        return self._data.loc[pokemon_id]['capture_dificulty']

    def get_reward(self, pokemon_id: int) -> int:
        return self._data.loc[pokemon_id]['reward']

    def get_name(self, pokemon_id: int) -> str:
        return self._data.loc[pokemon_id]['name']
