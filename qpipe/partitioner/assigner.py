import gurobipy as gp
from .._globals import MEM_UNIT as unit
from .utils import get_bit_layer_memory_map
def assign_bit_with_mem_constraints(T, available_bits, indicator, mem_constraints, model_mem_estimator, \
                                    store=False, verbose=False, file_path='bit_assignments.pkl'):
    L = len(T)
    s = get_bit_layer_memory_map(model_mem_estimator, available_bits)
    model = gp.Model()
    if not verbose:
        model.setParam("OutputFlag", 0)
    x = {}
    for l in range(L):
        for b in available_bits:
            x[(l, b)] = model.addVar(vtype=gp.GRB.BINARY)
    for l in range(L):
        model.addConstr(gp.quicksum(x[(l, b)] for b in available_bits) == 1)
    model.addConstr(gp.quicksum(s[shard, b] * x[(l, b)] for l, shard in enumerate(T) for b in available_bits) <= mem_constraints)
    model.setObjective(gp.quicksum(indicator[(l, b)] * x[(l, b)] for l in range(L) for b in available_bits), sense=gp.GRB.MINIMIZE)
    model.optimize()
    final_memory = sum(s[shard, b] * x[(l, b)].x for l, shard in enumerate(T) for b in available_bits if x[(l, b)].x > 0.5)

    final_assignments = {}
    for l in range(L):
        chosen_bits = [b for b in available_bits if x[(l, b)].x > 0.5]
        if verbose:
            print(f"Layer {l}: Chosen bits = {chosen_bits}")
        final_assignments[l] = chosen_bits[0]
    if verbose:
        print(f"Final memory = {final_memory} {unit}", mem_constraints)
    
    if store:
        import pickle
        with open(file_path, "wb") as f:
            pickle.dump(final_assignments, f)