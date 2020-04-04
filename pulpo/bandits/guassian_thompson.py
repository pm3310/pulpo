from typing import List, Dict

import numpy as np

from pulpo.bandits.dataclasses import GaussianArm, Feedback, Arm, BanditConfig
from pulpo.bandits.online_bandits import OnlineBandit
from pulpo.constants import fields


class GaussianThompsonBandit(OnlineBandit):
    _DEFAULT_N = 2
    _DEFAULT_REWARD_SUM = 2
    _DEFAULT_SQUARED_REWARD_SUM = 2

    def __init__(self, bandit_id: str, arms: List[GaussianArm]):
        super().__init__(bandit_id)
        self.bandit_id = bandit_id
        self.arms_dict: Dict[str, GaussianArm] = {arm.arm_id: arm for arm in arms}

    @classmethod
    def make_from_bandit_config(cls, config: BanditConfig):

        if config.priors:
            prior_n = config.priors.get(fields.N)
            prior_rewards_sum = config.priors.get(fields.REWARD_SUM)
            prior_squared_rewards_sum = config.priors.get(fields.SQUARED_REWARD_SUM)
        else:
            prior_n = GaussianThompsonBandit._DEFAULT_N
            prior_rewards_sum = GaussianThompsonBandit._DEFAULT_REWARD_SUM
            prior_squared_rewards_sum = GaussianThompsonBandit._DEFAULT_SQUARED_REWARD_SUM

        arms = [GaussianArm(arm_id=arm_id, n=prior_n, reward_sum=prior_rewards_sum,
                            squared_reward_sum=prior_squared_rewards_sum) for arm_id in config.arm_ids]

        return cls(config.bandit_id, arms)

    def choose(self, context: Dict[str, str] = None) -> Arm:
        arms = list(self.arms_dict.values())
        return max(arms, key=self._sample_score)

    def update(self, feedback: Feedback):
        arm = self.arms_dict[feedback.arm_id]
        arm.n += 1
        arm.reward_sum += feedback.reward
        arm.squared_reward_sum += pow(feedback.reward, 2)

    def reset(self):
        for arm in self.arms_dict.values():
            arm.n = 2
            arm.reward_sum = 2
            arm.squared_reward_sum = 2

    @staticmethod
    def _sample_score(arm: GaussianArm):
        mean = arm.reward_sum / arm.n
        sd = pow(arm.squared_reward_sum / arm.n - pow(arm.reward_sum / arm.n, 2), 0.5)
        return np.random.normal(mean, sd)
