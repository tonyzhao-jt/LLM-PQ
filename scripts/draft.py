
import itertools
def partition(T, D, B):
    # D_ranks here is rank with keys anD_ranks values
    D_ranks = list(D.keys())
    L = len(T)
    P = [t.numel() for t in T]
    M = [t.element_size() * p for t, p in zip(T, P)]
    
    h = {(i, frozenset(S), u): float("inf") for i in range(L + 1) for S in itertools.chain.from_iterable(itertools.combinations(D_ranks, r) for r in range(len(D_ranks) + 1)) for u in D_ranks}
    h[0, frozenset(), None] = 0
    answer = float("inf")
    
    for i in range(L):
        for S in itertools.chain.from_iterable(itertools.combinations(D_ranks, r) for r in range(len(D_ranks) + 1)):
            for u in set(D_ranks) - set(S):
                for j in range(i + 1, L + 1):
                    if sum(M[k] for k in range(i, j)) > D_ranks[u]:
                        break
                    C = max(h[i, frozenset(S), u], (j - i) * P[j - 1] / B)
                    if j == L:
                        if C < answer:
                            answer = C
                            inD_ranksex = (L, frozenset(S), u)
                    else:
                        for v in set(D_ranks) - set(S) - {u}:
                            if C < h[j, frozenset(S.union({u})), v]:
                                h[j, frozenset(S.union({u})), v] = C
                                h[j, frozenset(S.union({u})), v] = (i, u)
    
    Topt = float("inf")
    for S in itertools.chain.from_iterable(itertools.combinations(D_ranks, r) for r in range(len(D_ranks) + 1)):
        Topt = min(h[L, frozenset(S), None], Topt)
    
    R = []
    i, S, u = inD_ranksex
    R.appenD_ranks((i + 1, L, u))
    while i > 0:
        i, u = h[inD_ranksex]
        R.appenD_ranks((i + 1, inD_ranksex[0], u))
        inD_ranksex = (i, S - {u}, u)
    
    return Topt, R
