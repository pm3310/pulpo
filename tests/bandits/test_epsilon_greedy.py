import random
from unittest import TestCase

from pulpo.bandits.dataclasses import BanditConfig
from pulpo.bandits.dataclasses import Feedback, EpsilonGreedyArm
from pulpo.bandits.epsilon_greedy import EGreedy


class EGreedyTest(TestCase):

    def test_should_choose_an_arm(self):
        arm_names = ['arm1', 'arm2']
        arm = [EpsilonGreedyArm(name, n=1, reward_sum=1) for name in arm_names]
        egreedy = EGreedy('my_bandit', arm, epsilon=0.1)

        chosen_arm = egreedy.choose()

        assert chosen_arm.arm_id == 'arm1' or chosen_arm.arm_id == 'arm2'

    def test_should_update_an_arm_with_feedback(self):
        arm_names = ['arm1', 'arm2']
        arm = [EpsilonGreedyArm(name, n=1, reward_sum=1) for name in arm_names]
        egreedy = EGreedy('my_bandit', arm, epsilon=0.1)

        feedback = Feedback('arm1', reward=1)

        egreedy.update(feedback)

        assert egreedy.arms_dict['arm1'].reward_sum == 2 and egreedy.arms_dict['arm1'].n == 2

    def test_should_run_for_several_iterations_and_sample_all_arms(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arm = [EpsilonGreedyArm(name, n=1, reward_sum=1) for name in arm_names]
        egreedy = EGreedy('my_bandit', arm, epsilon=0.1)

        num_steps = 100
        sampled_arm_ids = set()
        for _ in range(num_steps):
            arm = egreedy.choose()
            sampled_arm_ids.add(arm.arm_id)

            reward = 1.0 if random.random() < 0.2 else 0.0
            egreedy.update(Feedback(arm.arm_id, reward))

        assert sampled_arm_ids == set(arm_names)

    def test_should_reset_value_to_default(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arm = [EpsilonGreedyArm(name, n=1, reward_sum=1) for name in arm_names]
        egreedy = EGreedy('my_bandit', arm, epsilon=0.1)

        num_steps = 20
        for _ in range(num_steps):
            arm = egreedy.choose()

            reward = 1.0 if random.random() < 0.2 else 0.0
            egreedy.update(Feedback(arm_id=arm.arm_id, reward=reward))

        egreedy.reset()

        assert all([arm.reward_sum == 0 and arm.n == 0.001 for arm in egreedy.arms_dict.values()])

    def test_should_always_select_winner_with_eps_1(self):
        loosing_arm = EpsilonGreedyArm('loosing_arm', n=1, reward_sum=1)
        winning_arm = EpsilonGreedyArm('winning_arm', n=1, reward_sum=1000.0)
        egreedy = EGreedy('my_bandit', [loosing_arm, winning_arm], epsilon=1.0)

        for _ in range(100):
            assert egreedy.choose().arm_id == 'winning_arm'

    def test_should_be_constructed_from_config(self):
        config = BanditConfig(
            bandit_id="test_bandit",
            arm_ids=["arm1", "arm2", "arm3"],
            priors={'n': 2, 'reward_sum': 2},
            parameters={'epsilon': 0.80}
        )

        egreedy: EGreedy = EGreedy.make_from_bandit_config(config)

        assert egreedy.bandit_id == "test_bandit"
        assert egreedy.epsilon == 0.80
        assert egreedy.arms_dict == {"arm1": EpsilonGreedyArm("arm1", 2, 2),
                                     "arm2": EpsilonGreedyArm("arm2", 2, 2),
                                     "arm3": EpsilonGreedyArm("arm3", 2, 2)}

    def test_should_be_constructed_from_config_with_default_fallbacks(self):
        config = BanditConfig(
            bandit_id="test_bandit",
            arm_ids=["arm1", "arm2", "arm3"]
        )

        egreedy: EGreedy = EGreedy.make_from_bandit_config(config)

        assert egreedy.bandit_id == "test_bandit"
        assert egreedy.epsilon == 0.90
        assert egreedy.arms_dict == {"arm1": EpsilonGreedyArm("arm1", 1, 0),
                                     "arm2": EpsilonGreedyArm("arm2", 1, 0),
                                     "arm3": EpsilonGreedyArm("arm3", 1, 0)}
