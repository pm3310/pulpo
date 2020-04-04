from typing import Dict, List

import numpy as np

from pulpo.bandits.dataclasses import Feedback, Arm, BetaArm, BanditConfig
from pulpo.bandits.online_bandits import OnlineBandit
from pulpo.constants import fields


class BetaThompsonBandit(OnlineBandit):
    _DEFAULT_N_REWARDS = 1
    _DEFAULT_N = 2

    def __init__(self, bandit_id: str, arms: List[BetaArm]):
        super().__init__(bandit_id)
        self.arms_dict: Dict[str, BetaArm] = {arm.arm_id: arm for arm in arms}

    @classmethod
    def make_from_bandit_config(cls, config: BanditConfig):

        if config.priors:
            prior_n = config.priors.get(fields.N)
            n_rewards = config.priors.get(fields.N_REWARDS)
        else:
            prior_n = BetaThompsonBandit._DEFAULT_N
            n_rewards = BetaThompsonBandit._DEFAULT_N_REWARDS

        arms = [BetaArm(arm_id=arm_id, n=prior_n, n_rewards=n_rewards) for arm_id in config.arm_ids]

        return cls(config.bandit_id, arms)

    def choose(self, context: Dict[str, str] = None) -> Arm:
        arms = list(self.arms_dict.values())
        return max(arms, key=self._get_score)

    def update(self, feedback: Feedback):
        arm = self.arms_dict[feedback.arm_id]
        arm.n += 1
        arm.n_rewards += feedback.reward

    def reset(self):
        for arm in self.arms_dict.values():
            arm.n = 2
            arm.n_rewards = 1

    def _get_score(self, arm: BetaArm):
        return np.random.beta(arm.n_rewards, arm.n - arm.n_rewards)
