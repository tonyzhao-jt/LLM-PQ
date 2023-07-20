def get_factors(x):
    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            # if i == 1:
            #     continue # don't include 1
            factors.append(i)
    return factors


def roundup_power2_divisions(n: int, k: int) -> int:
    """
    Rounds up the integer n to the nearest multiple of 2^k.

    Args:
        n (int): An integer to be rounded up.
        k (int): The power of 2 to which n should be rounded up.

    Returns:
        int: The closest multiple of 2^k that is greater than or equal to n.
    """
    power_of_two = 2 ** k
    return ((n - 1) // power_of_two + 1) * power_of_two