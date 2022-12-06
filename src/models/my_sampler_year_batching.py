import torch
from torch import Tensor
from torch.utils.data import Sampler

class ResumableSampler(Sampler):
    def __init__(self, data_source, batch_size, prev=None, mid_epoch = False, idx = 0) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
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
            self.prev = torch.IntTensor()
            year_start = 0
            df = self.data_source.dataset
            year = df.iloc[0, df.columns.get_loc("date")]
            for i in range(self.n):
                if i == self.n-1:
                    i += 1
                    cur_year = year + 1
                else:
                    cur_year = df.iloc[i, df.columns.get_loc("date")]
                if cur_year != year:
                    print(f"{year}: {year_start}")
                    year = cur_year
                    year_length = i-year_start
                    cur_perm = torch.randperm(year_length, generator=generator)[:year_length-(year_length % self.batch_size)] + int(year_start)
                    # print(cur_perm.type())
                    # print(year_length)
                    # print(len(cur_perm))
                    self.prev = torch.cat((self.prev, cur_perm))
                    year_start = i
                    # print(len(self.prev))
            batch_perm = torch.randperm(len(self.prev) // 32, generator=generator)
            batch_shuffled = torch.empty(len(self.prev), dtype=torch.int32)
            for i in range(len(batch_perm)):
                index = batch_perm[i]
                batch_shuffled[i*32:(i+1)*32] = self.prev[index*32:(index+1)*32]
            self.prev = batch_shuffled.tolist()
            yield from self.prev
    def __len__(self) -> int:
        return self.n