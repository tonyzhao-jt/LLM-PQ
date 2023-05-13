import qpipe
__LOCAL__TP__RANK__ = None
__LOCAL__PP__RANK__ = None
__PP__GROUP__RANKS__ = []
__TP__GROUP__RANKS__ = []
def get_local_tp_rank():
    return __LOCAL__TP__RANK__

def get_local_pp_rank():
    return __LOCAL__PP__RANK__

def set_local_tp_rank(rank):
    global __LOCAL__TP__RANK__
    __LOCAL__TP__RANK__ = rank

def set_local_pp_rank(rank):
    global __LOCAL__PP__RANK__
    __LOCAL__PP__RANK__ = rank

def get_prev_global_rank_pp():
    return __PP__GROUP__RANKS__[(__LOCAL__PP__RANK__ - 1) % len(__PP__GROUP__RANKS__)]

def get_next_global_rank_pp():
    return __PP__GROUP__RANKS__[(__LOCAL__PP__RANK__ + 1) % len(__PP__GROUP__RANKS__)]

def set_pp_group_ranks(ranks):
    for rank in ranks:
        assert isinstance(rank, int), "rank should be int"
        __PP__GROUP__RANKS__.append(rank)

def set_tp_group_ranks(ranks):
    for rank in ranks:
        assert isinstance(rank, int), "rank should be int"
        __TP__GROUP__RANKS__.append(rank)