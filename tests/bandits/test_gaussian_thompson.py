import random
from unittest import TestCase

from pulpo.bandits.dataclasses import Feedback, GaussianArm, BanditConfig
from pulpo.bandits.guassian_thompson import GaussianThompsonBandit


class GaussianThompsonBanditTest(TestCase):

    def test_should_choose_an_arm(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arms = [GaussianArm(name, 0.0001, 0, 1) for name in arm_names]

        gaussian_bandit = GaussianThompsonBandit('my_bandit', arms)

        chosen_arm = gaussian_bandit.choose()

        assert chosen_arm.arm_id == 'arm1' or chosen_arm.arm_id == 'arm2' or chosen_arm.arm_id == 'arm3'

    def test_should_update_an_arm_with_feedback(self):
        arm_names = ['arm1', 'arm2']
        arms = [GaussianArm(name, 0.0001, 0, 1) for name in arm_names]

        gaussian_bandit = GaussianThompsonBandit('my_bandit', arms)

        feedback = Feedback('arm2', reward=10)

        gaussian_bandit.update(feedback)

        assert gaussian_bandit.arms_dict['arm2'].reward_sum == 10
        assert gaussian_bandit.arms_dict['arm2'].squared_reward_sum == 101
        assert gaussian_bandit.arms_dict['arm2'].n == 1.0001

    def test_should_run_for_several_iterations_and_sample_all_arms(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arms = [GaussianArm(name, n=5, reward_sum=5, squared_reward_sum=5) for name in arm_names]

        gaussian_bandit = GaussianThompsonBandit('my_bandit', arms)

        num_steps = 1000
        sampled_arm_ids = set()
        for _ in range(num_steps):
            arm = gaussian_bandit.choose()
            sampled_arm_ids.add(arm.arm_id)

            reward = 1.0 if random.random() < 0.2 else 0.0
            gaussian_bandit.update(Feedback(arm.arm_id, reward))

        assert sampled_arm_ids == set(arm_names)

    def test_should_reset_value_to_default(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arms = [GaussianArm(name, 2, 2, 2) for name in arm_names]

        gaussian_bandit = GaussianThompsonBandit('my_bandit', arms)
        num_steps = 20
        for _ in range(num_steps):
            arm = gaussian_bandit.choose()

            reward = 1.0 if random.random() < 0.2 else 0.0
            gaussian_bandit.update(Feedback(arm_id=arm.arm_id, reward=reward))

        gaussian_bandit.reset()

        assert all([arm.squared_reward_sum == 2 and arm.reward_sum == 2 and arm.n == 2 for arm in
                    gaussian_bandit.arms_dict.values()])

    def test_should_always_select_winner_with_obvious(self):
        winning_arm = GaussianArm("winning_arm", 200, 200, 200)
        losing_arm = GaussianArm("losing_arm", 200, -200, 200)

        gaussian_bandit = GaussianThompsonBandit('my_bandit', [winning_arm, losing_arm])

        for _ in range(100):
            assert gaussian_bandit.choose().arm_id == "winning_arm"

    def test_should_be_constructed_from_config(self):
        config = BanditConfig(
            bandit_id="test_bandit",
            arm_ids=["arm1", "arm2", "arm3"],
            priors={'n': 3, 'reward_sum': 3, 'squared_reward_sum': 4}
        )

        gaussian_thompson: GaussianThompsonBandit = GaussianThompsonBandit.make_from_bandit_config(config)

        assert gaussian_thompson.bandit_id == "test_bandit"
        assert gaussian_thompson.arms_dict == {"arm1": GaussianArm("arm1", 3, 3, 4),
                                               "arm2": GaussianArm("arm2", 3, 3, 4),
                                               "arm3": GaussianArm("arm3", 3, 3, 4)}

    def test_should_be_constructed_from_config_with_default_fallbacks(self):
        config = BanditConfig(
            bandit_id="test_bandit",
            arm_ids=["arm1", "arm2", "arm3"]
        )

        gaussian_thompson: GaussianThompsonBandit = GaussianThompsonBandit.make_from_bandit_config(config)

        assert gaussian_thompson.bandit_id == "test_bandit"
        assert gaussian_thompson.arms_dict == {"arm1": GaussianArm("arm1", 2, 2, 2),
                                               "arm2": GaussianArm("arm2", 2, 2, 2),
                                               "arm3": GaussianArm("arm3", 2, 2, 2)}
