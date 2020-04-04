import json
from typing import List
from unittest import TestCase

from pulpo.bandit_factory import BanditFactory
from pulpo.bandits.dataclasses import EpsilonGreedyArm
from pulpo.bandits.epsilon_greedy import EGreedy
from pulpo.pulpo import Pulpo


class PulpoTest(TestCase):

    def test_should_be_instantiated_with_internal_map(self):
        arm_names = ["arm1", "arm2", "amr3"]
        arms = [EpsilonGreedyArm(name, 1, 0) for name in arm_names]

        bandit1 = EGreedy("bandit1", arms, epsilon=0.9)
        bandit2 = EGreedy("bandit2", arms, epsilon=0.9)

        pulpo = Pulpo([bandit1, bandit2])

        assert list(pulpo.bandits.values()) == [bandit1, bandit2]

    def test_should_choose_an_arm_per_bandit(self):
        arm_names = ["arm1", "arm2", "amr3"]
        arms = [EpsilonGreedyArm(name, 1, 0) for name in arm_names]

        bandit1 = EGreedy("bandit1", arms, epsilon=0.9)
        bandit2 = EGreedy("bandit2", arms, epsilon=0.9)

        pulpo = Pulpo([bandit1, bandit2])

        assert pulpo.choose('bandit1') in ["arm1", "arm2", "amr3"]
        assert pulpo.choose('bandit2') in ["arm1", "arm2", "amr3"]

    def test_should_be_updated_with_feedback(self):
        arm_names = ["arm1", "arm2", "amr3"]
        arms = [EpsilonGreedyArm(name, 1, 0) for name in arm_names]

        bandit1 = EGreedy("bandit1", arms, epsilon=0.9)
        bandit2 = EGreedy("bandit2", arms, epsilon=0.9)

        pulpo = Pulpo([bandit1, bandit2])

        pulpo.update("bandit2", "arm2", 100)

        bandit2_arm2 = pulpo.bandits['bandit2'].arms_dict['arm2']
        assert bandit2_arm2.reward_sum == 100 and bandit2_arm2.n == 2

    def test_should_reset(self):
        arm_names = ["arm1", "arm2", "amr3"]
        arms = [EpsilonGreedyArm(name, 1, 0) for name in arm_names]

        bandit1 = EGreedy("bandit1", arms, epsilon=0.9)
        bandit2 = EGreedy("bandit2", arms, epsilon=0.9)

        pulpo = Pulpo([bandit1, bandit2])

        pulpo.update("bandit2", "arm2", 100)

        pulpo.reset("bandit2")

        bandit2_arm2 = pulpo.bandits['bandit2'].arms_dict['arm2']
        assert bandit2_arm2.reward_sum == 0 and bandit2_arm2.n == 0.001

    def test_should_be_created_from_json(self):
        bandit_types: List[str] = list(BanditFactory.MAPPING.keys())

        json_string = '[' + ", ".join([PulpoTest._get_default_config(type) for type in bandit_types]) + ']'

        pulpo = Pulpo.make_from_json(json_string)

        assert len(pulpo.bandits) == len(BanditFactory.MAPPING)

    @staticmethod
    def _get_default_config(bandit_type: str):
        default_values = {"bandit_id": "test_bandit_" + bandit_type, "bandit_type": bandit_type,
                          "arm_ids": ["arm1", "arm2", "arm3"]}

        return json.dumps(default_values)
