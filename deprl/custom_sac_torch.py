import torch

from deprl.vendor.tonic.torch import agents

class TunedSAC(agents.SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)