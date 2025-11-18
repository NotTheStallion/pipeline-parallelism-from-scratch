from collections import defaultdict
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import plot_memory_and_schedule, precedes


p = 4                   # @param GPUs / stages
m = p-1                 # @param microbatches per batch
T_comm = 0.0            # @param inter-stage comm time
M_B, M_W = 25, 10       # @param memory deltas (GB)
gpu_mem_limit = p*M_B   # @param GPU memory limit (GB)
alpha = 1               # @param factor for teacher forwards
extra_teach_f = p-1     # @param number of microbatches of the "next batch" teacher forwards


T = {}
for stage in range(1, p+1):
    for mb in range(1, m+1):
        T[(stage, mb, 'F_S')] = 1
        T[(stage, mb, 'B')]   = 1
        T[(stage, mb, 'W')]   = 1

# Next batch teacher forwards
for stage in range(1, p+1):
    for mb in range(1, extra_teach_f+m+1):
        T[(stage, mb, 'F_T')] = alpha * T[(1, 1, 'F_S')]

def has(stage, mb, op):
    return (stage, mb, op) in T

mdl = pulp.LpProblem("ZB_ILP_fixed_ordered_next_batch_teacher_only", pulp.LpMinimize)
S = {k: pulp.LpVariable(f"S_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
E = {k: pulp.LpVariable(f"E_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
Z = pulp.LpVariable("Z", lowBound=0)

# Objective: minimize (per-stage last W of first batch) – (earliest first-next-batch forward start)
for stage in range(1, p+1):
    # earliest of F_S1 and F_T1 (first-batch only)
    W0 = pulp.LpVariable(f"W_{stage}", lowBound=0)
    mdl += W0 <= S[(stage, 1, 'F_S')]
    mdl += W0 <= S[(stage, p, 'F_T')]
    
    
    # max W_m and F_T_2m
    W1 = pulp.LpVariable(f"W1_{stage}", lowBound=0)
    mdl += W1 >= E[(stage, m, 'W')]
    mdl += W1 >= E[(stage, extra_teach_f+m, 'F_T')]

    mdl += Z >= W1 - W0
mdl += Z

for k, dur in T.items():
    mdl += E[k] == S[k] + dur

# Intra-microbatch chaining (first batch only)
for stage in range(1, p+1):
    for mb in range(1, m+1):
        mdl += S[(stage, mb, 'B')] >= E[(stage, mb, 'F_S')] + T_comm
        mdl += S[(stage, mb, 'B')] >= E[(stage, mb, 'F_T')] + T_comm
        mdl += S[(stage, mb, 'W')] >= E[(stage, mb, 'B')] + T_comm

# Microbatch order per op on each stage
for stage in range(1, p+1):
    # F_T must be ordered across 1..2m (when both exist)
    for j in range(2, extra_teach_f+m+1):
        if has(stage, j, 'F_T') and has(stage, j-1, 'F_T'):
            mdl += S[(stage, j, 'F_T')] >= E[(stage, j-1, 'F_T')] + T_comm
    # F_S, B, W are only for first batch (1..m)
    for op in ['F_S', 'B', 'W']:
        for j in range(2, m+1):
            mdl += S[(stage, j, op)] >= E[(stage, j-1, op)] + T_comm

# Inter-stage forward deps
for stage in range(2, p+1):
    # F_S flows only for first batch
    for mb in range(1, m+1):
        mdl += S[(stage, mb, 'F_S')] >= E[(stage-1, mb, 'F_S')] + T_comm
    # F_T flows for both batches (guard existence)
    for mb in range(1, extra_teach_f+m+1):
        if has(stage-1, mb, 'F_T') and has(stage, mb, 'F_T'):
            mdl += S[(stage, mb, 'F_T')] >= E[(stage-1, mb, 'F_T')] + T_comm

# Backward dependency (first batch only)
for stage in range(1, p):
    for mb in range(1, m+1):
        mdl += S[(stage, mb, 'B')] >= E[(stage+1, mb, 'B')] + T_comm


M = 1e5
y = {}


for stage in range(1, p+1):
    tasks_on_stage = [task for task in T if task[0] == stage]
    for a in tasks_on_stage:
        for b in tasks_on_stage:
            if a == b:
                continue
            if (a, b) not in y and (b, a) not in y:
                y[(a, b)] = pulp.LpVariable(f"y_{a}_{b}", lowBound=0, upBound=1, cat="Binary")
                # Mutually exclusive ordering on end-times (works since E = S + dur)
                mdl += E[a] >= E[b] + T[a] - M * precedes(a, b, y)
                mdl += E[b] >= E[a] + T[b] - M * precedes(b, a, y)

delta_mem = {'F_S': M_B, 'F_T':0, 'B': M_W - M_B, 'W': -M_W}

for stage in range(1, p+1):
    tasks_stage = [t for t in T if t[0] == stage]
    for b in tasks_stage:
        # capacity at every completion instant E[b]
        mdl += pulp.lpSum(delta_mem[a[2]] * precedes(a, b, y) for a in tasks_stage) <= gpu_mem_limit

mdl.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*60*4)) # type: ignore
print("Status:", pulp.LpStatus[mdl.status])
print("Objective (Z):", pulp.value(Z))

schedule = defaultdict(list)
for task in sorted(T):
    s = float(pulp.value(S[task])) # type: ignore
    e = float(pulp.value(E[task])) # type: ignore
    schedule[task[0]].append((s, e, task[1], task[2]))

plot_memory_and_schedule(schedule, T, delta_mem, p, m, filename_prefix="zbts_p-1_ilp")
