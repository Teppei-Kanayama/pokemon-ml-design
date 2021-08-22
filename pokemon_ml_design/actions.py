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
    Action(label='モンスターボールを投げる', performance=50, cost=100),
    Action(label='スーパーボールを投げる', performance=100, cost=500),
    Action(label='ハイパーボールを投げる', performance=200, cost=2000),
    Action(label='マスターボールを投げる', performance=math.inf, cost=10000),
]
