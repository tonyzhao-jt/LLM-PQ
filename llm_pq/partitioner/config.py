# store the generation configs
import dataclasses

@dataclasses.dataclass
class GenerationConfig:
    global_bz: int
    micro_bz: int
    # prompt length, generated sequence length
    s: int
    n: int 

gen_config = GenerationConfig(global_bz=16, micro_bz=4, s=512, n=100)