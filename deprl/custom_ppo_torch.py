import torch

from deprl import custom_torso
from deprl.vendor.tonic.torch import agents, updaters


class TunedPPO(agents.PPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
