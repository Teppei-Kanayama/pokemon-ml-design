from typing import List
from dataclasses import dataclass
import math


@dataclass
class Action:
    label: str
    label_en: str
    performance: int
    cost: int
    color: str
    marker: str


ACTIONS: List[Action] = [
    Action(label='逃げる', label_en='RUN AWAY', performance=-math.inf, cost=0, color='black', marker='o'),
    Action(label='モンスターボールを投げる', label_en='THROW MONSTER-BALL', performance=50, cost=100, color='red', marker='s'),
    Action(label='スーパーボールを投げる', label_en='THROW SUPER-BALL', performance=100, cost=500, color='blue', marker='v'),
    Action(label='ハイパーボールを投げる', label_en='THROW HYPER-BALL', performance=200, cost=2000, color='orange', marker='^'),
    Action(label='マスターボールを投げる', label_en='THROW MASTER-BALL', performance=math.inf, cost=10000, color='violet', marker='x'),
]
