import random as rand
from typing import List, Dict

from pulpo.bandits.dataclasses import BanditConfig
from pulpo.bandits.dataclasses import EpsilonGreedyArm, Arm, Feedback
from pulpo.bandits.online_bandits import OnlineBandit
from pulpo.constants import fields


class EGreedy(OnlineBandit):
    _DEFAULT_N = 1
    _DEFAULT_REWARD_SUM = 0
    _DEFAULT_EPSILON = 0.90
    """
    Implementation of EGreedy algorithm as described in Section 2 of book:

    Reinforcement Learning: An Introduction (Version 2)
    Richard S. Sutton and Andrew G. Barto
    """

    def __init__(self, bandit_id: str, arms: List[EpsilonGreedyArm], epsilon):
        super().__init__(bandit_id)
        """
        Constructor of EGreedy

        :param arm_ids: [List[str]], list of arm ids to instantiate.
        :param epsilon: [float, default=0.1], epsilon value in range (0.0, 1.0) for exploration
        """
        self.epsilon: float = epsilon
        self.arms_dict: Dict[str, EpsilonGreedyArm] = {arm.arm_id: arm for arm in arms}

    @classmethod
    def make_from_bandit_config(cls, config: BanditConfig):

        if config.priors:
            n = config.priors[fields.N]
            reward_sum = config.priors[fields.REWARD_SUM]
        else:
            n = EGreedy._DEFAULT_N
            reward_sum = EGreedy._DEFAULT_REWARD_SUM

        if config.parameters:
            epsilon = config.parameters[fields.EPSILON]
        else:
            epsilon = EGreedy._DEFAULT_EPSILON

        arms = [EpsilonGreedyArm(arm_id, n, reward_sum) for arm_id in config.arm_ids]

        return cls(config.bandit_id, arms, epsilon)

    def reset(self):
        for arm in self.arms_dict.values():
            arm.n = 0.001
            arm.reward_sum = 0

    def choose(self, context=None) -> Arm:

        if rand.random() >= self.epsilon:
            return rand.choice(list(self.arms_dict.values()))
        else:
            return max(self.arms_dict.values(), key=lambda arm: arm.reward_sum / arm.n)

    def update(self, feedback: Feedback):
        arm = self.arms_dict[feedback.arm_id]

        arm.n += 1
        arm.reward_sum += feedback.reward
