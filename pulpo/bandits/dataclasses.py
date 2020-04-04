from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class Arm:
    arm_id: str


@dataclass
class EpsilonGreedyArm(Arm):
    n: float
    reward_sum: float


@dataclass
class GaussianArm(Arm):
    n: float
    reward_sum: float
    squared_reward_sum: float


@dataclass
class BetaArm(Arm):
    n: float
    n_rewards: float


@dataclass
class Feedback:
    arm_id: str
    reward: float
    payload: str = None


@dataclass
class BanditConfig:
    bandit_id: str
    arm_ids: List[str]
    priors: Optional[Dict[str, float]] = None
    parameters: Optional[Dict[str, float]] = None
