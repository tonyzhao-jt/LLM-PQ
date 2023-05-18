def get_available_bits_pair(available_bits):
    available_bits = list(set(available_bits))
    BITs = [
        (i, j) for i in available_bits for j in available_bits
    ]
    return BITs