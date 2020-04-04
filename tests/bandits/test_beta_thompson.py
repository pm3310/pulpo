import random
from unittest import TestCase

from pulpo.bandits.beta_thompson import BetaThompsonBandit
from pulpo.bandits.dataclasses import Feedback, BetaArm, BanditConfig


class BetaThompsonBanditTest(TestCase):

    def test_should_choose_an_arm(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arms = [BetaArm(name, n=2, n_rewards=1) for name in arm_names]

        beta_bandit = BetaThompsonBandit('my_bandit', arms)

        chosen_arm = beta_bandit.choose()

        assert chosen_arm.arm_id == 'arm1' or chosen_arm.arm_id == 'arm2' or chosen_arm.arm_id == 'arm3'

    def test_should_update_an_arm_with_feedback(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arms = [BetaArm(name, n=2, n_rewards=1) for name in arm_names]

        beta_bandit = BetaThompsonBandit('my_bandit', arms)

        feedback = Feedback('arm2', reward=1)

        beta_bandit.update(feedback)

        assert beta_bandit.arms_dict['arm2'].n_rewards == 2
        assert beta_bandit.arms_dict['arm2'].n == 3

    def test_should_run_for_several_iterations_and_sample_all_arms(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arms = [BetaArm(name, n=2, n_rewards=1) for name in arm_names]

        beta_bandit = BetaThompsonBandit('my_bandit', arms)

        num_steps = 1000
        sampled_arm_ids = set()
        for _ in range(num_steps):
            arm = beta_bandit.choose()
            sampled_arm_ids.add(arm.arm_id)

            reward = 1.0 if random.random() < 0.2 else 0.0
            beta_bandit.update(Feedback(arm.arm_id, reward))

        assert sampled_arm_ids == set(arm_names)

    def test_should_reset_value_to_default(self):
        arm_names = ['arm1', 'arm2', 'arm3']
        arms = [BetaArm(name, n=2, n_rewards=1) for name in arm_names]

        beta_bandit = BetaThompsonBandit('my_bandit', arms)
        num_steps = 20
        for _ in range(num_steps):
            arm = beta_bandit.choose()

            reward = 1.0 if random.random() < 0.2 else 0.0
            beta_bandit.update(Feedback(arm_id=arm.arm_id, reward=reward))

        beta_bandit.reset()

        assert all([arm.n_rewards == 1 and arm.n == 2 for arm in beta_bandit.arms_dict.values()])

    def test_should_always_select_winner_with_obvious(self):
        winning_arm = BetaArm("winning_arm", 100000, 99999)
        losing_arm = BetaArm("losing_arm", 100000, 1)

        beta_bandit = BetaThompsonBandit('my_bandit', [winning_arm, losing_arm])

        for i in range(100):
            assert beta_bandit.choose().arm_id == "winning_arm"

    def test_should_be_constructed_from_config(self):
        config = BanditConfig(
            bandit_id="test_bandit",
            arm_ids=["arm1", "arm2", "arm3"],
            priors={'n': 3, 'n_rewards': 2}
        )

        gaussian_thompson: BetaThompsonBandit = BetaThompsonBandit.make_from_bandit_config(config)

        assert gaussian_thompson.bandit_id == "test_bandit"
        assert gaussian_thompson.arms_dict == {"arm1": BetaArm("arm1", 3, 2),
                                               "arm2": BetaArm("arm2", 3, 2),
                                               "arm3": BetaArm("arm3", 3, 2)}

    def test_should_be_constructed_from_config_with_default_fallbacks(self):
        config = BanditConfig(
            bandit_id="test_bandit",
            arm_ids=["arm1", "arm2", "arm3"]
        )

        gaussian_thompson: BetaThompsonBandit = BetaThompsonBandit.make_from_bandit_config(config)

        assert gaussian_thompson.bandit_id == "test_bandit"
        assert gaussian_thompson.arms_dict == {"arm1": BetaArm("arm1", 2, 1),
                                               "arm2": BetaArm("arm2", 2, 1),
                                               "arm3": BetaArm("arm3", 2, 1)}
