import torch
from torch import Tensor
from torch.utils.data import Sampler

class ResumableSampler(Sampler):
    def __init__(self, data_source, prev=None, mid_epoch = False, idx = 0) -> None:
        self.data_source = data_source
        self.set_mid_epoch = False
        self.mid_epoch = mid_epoch
        self.prev = prev
        self.idx = idx
        self.n = len(self.data_source)
        self.generator = None
        print("----------Sampler init----------")
        print(f"mid epoch = {self.mid_epoch}")

    def __iter__(self):
        print("----------Sampler iter----------")
        if self.mid_epoch:
            print("_____mid epoch_____")
            # print(f"{self.prev}")
            # print(f"self.idx = {self.idx}")
            # print(f"{self.prev[self.idx:]}")
            yield from self.prev[self.idx:]
        else:
            print("_____not mid epoch_____")
            if self.generator is None:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            else:
                generator = self.generator
            self.prev = torch.randperm(self.n, generator=generator).tolist()
            yield from self.prev
    def __len__(self) -> int:
        return self.n