import json
from typing import List
from unittest import TestCase

from pulpo.bandit_factory import BanditFactory
from pulpo.bandits.dataclasses import EpsilonGreedyArm
from pulpo.bandits.epsilon_greedy import EGreedy
from pulpo.bandits.online_bandits import OnlineBandit


class BanditFactoryTest(TestCase):

    def test_should_build_from_json(self):
        json_string = """
        [
            {
                "bandit_id" : "test_bandit",
                "bandit_type" : "epsilon_greedy",
                "arm_ids" : ["arm1", "arm2", "arm3"],
                "priors" : {"n": 2, "reward_sum": 2},
                "parameters" : {"epsilon": 0.80}
            }
        ]
        """
        factory = BanditFactory()

        bandit_list: List[OnlineBandit] = factory.make_bandits_list(json_string)
        new_bandit: EGreedy = bandit_list[0]

        assert new_bandit.bandit_id == "test_bandit"
        assert new_bandit.epsilon == 0.80
        assert new_bandit.arms_dict == {
            "arm1": EpsilonGreedyArm("arm1", 2, 2),
            "arm2": EpsilonGreedyArm("arm2", 2, 2),
            "arm3": EpsilonGreedyArm("arm3", 2, 2)
        }

    def test_should_build_from_json_with_optional_parameters_omited(self):
        json_string = """
        [
            {
                "bandit_id" : "test_bandit",
                "bandit_type" : "epsilon_greedy",
                "arm_ids" : ["arm1", "arm2", "arm3"]
            }
        ]
        """
        factory = BanditFactory()

        bandit_list: List[OnlineBandit] = factory.make_bandits_list(json_string)
        new_bandit: EGreedy = bandit_list[0]

        assert new_bandit.bandit_id == "test_bandit"
        assert new_bandit.epsilon == 0.90
        assert new_bandit.arms_dict == {
            "arm1": EpsilonGreedyArm("arm1", 1, 0),
            "arm2": EpsilonGreedyArm("arm2", 1, 0),
            "arm3": EpsilonGreedyArm("arm3", 1, 0)
        }

    def test_should_instantiate_all_types_of_bandit_with_default_values(self):
        bandit_types: List[str] = list(BanditFactory.MAPPING.keys())

        json_string = '[' + ", ".join([BanditFactoryTest._get_default_config(type) for type in bandit_types]) + ']'

        factory = BanditFactory()

        assert len(factory.make_bandits_list(json_string)) == len(BanditFactory.MAPPING)

    @staticmethod
    def _get_default_config(bandit_type: str):
        default_values = {"bandit_id": "test_bandit_" + bandit_type, "bandit_type": bandit_type,
                          "arm_ids": ["arm1", "arm2", "arm3"]}

        return json.dumps(default_values)
