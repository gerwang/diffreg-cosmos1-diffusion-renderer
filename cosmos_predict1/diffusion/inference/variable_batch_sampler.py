from torch.utils.data import Sampler

# 自定义 BatchSampler
class VariableBatchSampler(Sampler):
    def __init__(self, dataset_size: int, batch_sizes: list[int]):
        self.dataset_size = dataset_size
        self.batch_sizes = batch_sizes

    def __iter__(self):
        idx = 0
        for bsz in self.batch_sizes:
            if idx >= self.dataset_size:
                break
            end = min(idx + bsz, self.dataset_size)
            yield list(range(idx, end))
            idx = end

    def __len__(self):
        # batch 数量（不一定与 batch_sizes 长度相同，若超出数据范围）
        total = 0
        idx = 0
        for bsz in self.batch_sizes:
            if idx >= self.dataset_size:
                break
            total += 1
            idx += bsz
        return total