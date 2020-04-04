from typing import List, Dict

from pulpo.bandit_factory import BanditFactory
from pulpo.bandits.dataclasses import Feedback
from pulpo.bandits.online_bandits import OnlineBandit


class Pulpo:
    def __init__(self, bandits: List[OnlineBandit]):
        """
        Pulpo constructor.

        The objective of this class is to manage the bandit campaign.

        :param bandits: List[OnlineBandit], List of bandits that will be managed.

        """
        self.bandits: map[str, OnlineBandit] = {bandit.bandit_id: bandit for bandit in bandits}

    @classmethod
    def make_from_json(cls, configuration: str):
        bandits: List[OnlineBandit] = BanditFactory.make_bandits_list(configuration)
        return cls(bandits)

    def reset(self, bandit_id: str):
        """
        Resets state of bandit strategy
        """
        self.bandits[bandit_id].reset()

    def choose(self, bandit_id: str, context: Dict[str, str] = None) -> str:
        """
        Chooses an arm

        :param context: [dict, default=None], context of the request. Only discrete context values
        are supported of type str. Example:

        {
            "context_a": "value_a1",
            "context_b: "value_b3",
            "context_c: "value_c2"
        }

        :return: [str], an arm name
        """
        arm = self.bandits[bandit_id].choose(context)
        return arm.arm_id

    def update(self, bandit_id: str, arm_id: str, reward: float, payload: str = None) -> str:
        """
        Updates bandit strategy given the feedback

        :param bandit_id: [str], bandit id
        :param arm_id: [str], arm name
        :param reward: [float], reward of the chosen arm name
        :param context: [dict, default=None], context used when choose was called.
        """
        bandit: OnlineBandit = self.bandits[bandit_id]
        bandit.update(Feedback(arm_id, reward, payload))
