from abc import ABCMeta, abstractmethod
from typing import Dict

from pulpo.bandits.dataclasses import Arm, Feedback, BanditConfig


class OnlineBandit(metaclass=ABCMeta):
    """
    Base abstract class to inherit from for Online MAB implementations
    """
    def __init__(self, bandit_id: str):
        self.bandit_id: str = bandit_id

    @classmethod
    @abstractmethod
    def make_from_bandit_config(cls, config: BanditConfig):
        """
        Alternate constructor that uses a bandit configuration to instatiate the class.
        :param config: BanditConfig, this the dataclass which contains the values necessary to configure the bandit
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets state of the algorithm
        """
        pass

    @abstractmethod
    def choose(self, context: Dict[str, str] = None) -> Arm:
        """
        Chooses an arm id

        :param context: [dict, default=None], context of the request. Only discrete context values
        are supported of type str. Example:

        {
            "context_a": "value_a1",
            "context_b: "value_b3"
        }

        :return: [Arm], arm dataclass with all arm info.
        """
        pass

    @abstractmethod
    def update(self, feedback: Feedback):
        """
        Updates algorithm given the feedback

        :param feeback: [Feedback], dataclass containing the armid and reward
        """
        pass

    @property
    def path(self):
        raise NotImplementedError
