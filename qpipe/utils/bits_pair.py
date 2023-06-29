def get_available_bits_pair(available_bits):
    available_bits = list(set(available_bits))
    # we abandon the pairs of different bits due to some intra optimization for bitsandbytes
    # which resulting in quite different performance in profiling and real execution
    # BITs = [
    #     (i, j) for i in available_bits for j in available_bits
    # ]
    BITs = [
        (i, i) for i in available_bits
    ]
    # sort
    BITs = sorted(BITs, key=lambda x: str(x[0]))
    return BITs

# get decouple dict, as we allowed different bit for MLP and attention
def get_available_bits_pair_decoupled(available_bits: list) -> dict:
    available_bits = list(set(available_bits))
    # we abandon the pairs of different bits due to some intra optimization for bitsandbytes
    # which resulting in quite different performance in profiling and real execution
    BITs = [
        (i, j) for i in available_bits for j in available_bits
    ]
    # sort
    BITs = sorted(BITs, key=lambda x: str(x[0]))
    return BITs