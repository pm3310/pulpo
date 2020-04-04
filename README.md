**Build Status**

master: [![Build Status](https://travis-ci.org/pm3310/pulpo.svg?branch=master)](https://travis-ci.org/pm3310/pulpo)

# pulpo

A library and sdk for non-contextual and contextual Multi-Armed-Bandit (MAB) algorithms for multiple use cases. The sdk version enables you to deploy it on AWS. 

## Installation

### Prerequisites

pulpo requires Python 3.6+

### Install pulpo

At the command line:

    pip install pulpo

## How To

### Library

Pulpo can be used as a library to instantiate and run online MABs. The core of the library are the bandit implementations under `pulpo/badnits` module. The super class of them is `OnlineBandit` if you wish to implement your own MAB implementation.

Currently, the following MAB implementations are available:
- Epsilon Greedy (`EGreedy`)

The usage of this library starts from `Pulpo` class. For example, let's say that we want to run 1 online bandit with 3 arms using `EGreedy`, then:
```Python
from pulpo.bandits.dataclasses import EpsilonGreedyArm
from pulpo.bandits.epsilon_greedy import EGreedy
from pulpo.pulpo import Pulpo

arm_names = ["article1", "article2", "article3"]
arms = [EpsilonGreedyArm(name, 1, 0) for name in arm_names]  # priors for n=1 and sum=0, i.e. steps and total reward so far

bandit = EGreedy("article_recommendation", arms, epsilon=0.9)

pulpo = Pulpo([bandit])  # Instantiate Pulpo

arm_id = pulpo.choose(bandit.bandit_id)  # to get an arm decision

# get some feedback for the arm decision and pass it to back

feedback = 1.0
pulpo.update(bandit.bandit_id, arm_id, feedback)
```

Alternatively, the `Pulpo` class can be instantiated in the following manner:
```Python
import json

from pulpo.pulpo import Pulpo

bandit_id = "article_recommendation"
config = config = [{"bandit_id": bandit_id, "bandit_type": "epsilon_greedy", "arm_ids": ["article1", "article2", "article3"]}]
pulpo = Pulpo.make_from_json(json.dumps(config))  # Instantiate Pulpo

arm_id = pulpo.choose(bandit_id)  # to get an arm decision

# get some feedback for the arm decision and pass it to back

feedback = 1.0
pulpo.update(bandit_id, arm_id, feedback)
```

### AWS SDK

Pulpo can be used as an sdk to deploy and run MABs on AWS. Soon...