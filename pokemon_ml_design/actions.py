from typing import List
from dataclasses import dataclass

import math


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
