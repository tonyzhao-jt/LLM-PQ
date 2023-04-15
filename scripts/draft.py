def find_pairs(n):
    pairs = []
    for i in range(1, n+1):
        if n % i == 0:
            pairs.append((i, n//i))
    return pairs

print(find_pairs(16))