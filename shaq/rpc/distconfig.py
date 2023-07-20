from dataclasses import dataclass
@dataclass
class DistConfig:
    local_rank: int
    rank: int
    group_rank: int
    world_size: int
    ngpus: int

    def __init__(self, local_rank, rank, group_rank, world_size, ngpus):
        self.local_rank = local_rank
        self.rank = rank
        self.group_rank = group_rank
        self.world_size = world_size
        self.ngpus = ngpus