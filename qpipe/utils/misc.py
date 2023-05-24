def get_factors(x):
    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            # if i == 1:
            #     continue # don't include 1
            factors.append(i)
    return factors