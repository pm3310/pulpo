import json
from typing import List, Dict

from pulpo.bandits.beta_thompson import BetaThompsonBandit
from pulpo.bandits.dataclasses import BanditConfig
from pulpo.bandits.epsilon_greedy import EGreedy
from pulpo.bandits.guassian_thompson import GaussianThompsonBandit
from pulpo.bandits.online_bandits import OnlineBandit
from pulpo.constants import fields


class BanditFactory:
    MAPPING: Dict[str, OnlineBandit] = {
        'epsilon_greedy': EGreedy,
        'gaussian_thompson': GaussianThompsonBandit,
        'beta_thompson': BetaThompsonBandit}

    @staticmethod
    def make_bandits_list(instructions: str) -> List[OnlineBandit]:
        instructions = json.loads(instructions)
        return [BanditFactory.make_bandit(instruction) for instruction in instructions]

    @staticmethod
    def make_bandit(instruction) -> OnlineBandit:
        config: BanditConfig = BanditFactory.parse_bandit_config(instruction)
        bandit_type = instruction.get(fields.BANDIT_TYPE)
        bandit_class = BanditFactory.MAPPING.get(bandit_type)
        return bandit_class.make_from_bandit_config(config)

    @staticmethod
    def parse_bandit_config(bandit_config) -> BanditConfig:
        bandit_id: str = bandit_config[fields.BANDIT_ID]
        arm_ids = bandit_config[fields.ARM_IDS]
        if bandit_config.get(fields.PRIORS):
            priors = bandit_config.get(fields.PRIORS)
        else:
            priors = None
        if bandit_config.get(fields.PARAMETERS):
            parameters = bandit_config.get(fields.PARAMETERS)
        else:
            parameters = None

        return BanditConfig(bandit_id, arm_ids, priors, parameters)
