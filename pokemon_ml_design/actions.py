from typing import List
from dataclasses import dataclass

import math

# TODO: classとして持つには大げさか？
# class Ball:
#     def __init__(self, ball_type: str) -> None:
#         self._ball_coefficient_mapper = dict(
#             monster=0,
#             super=100,
#             hyper=300,
#             master=None,
#         )
#         self.ball_type = ball_type
#
#     def get_coefficient(self) -> int:
#         return self._ball_coefficient_mapper[self.ball_type]
#
# # 100倍した方が迫力が出るかもしれない
# ball_price = {0: 0, 1: 50, 2: 100, 3: 200, 4: 500}

@dataclass
class Action:
    label: str
    performance: int
    cost: int


ACTIONS: List[Action] = [
    Action(label='逃げる', performance=-math.inf, cost=0),
    Action(label='モンスターボールを投げる', performance=0, cost=50),
    Action(label='スーパーボールを投げる', performance=100, cost=100),
    Action(label='ハイパーボールを投げる', performance=300, cost=200),
    Action(label='マスターボールを投げる', performance=math.inf, cost=500),
]
