from collections import defaultdict
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import plot_memory_and_schedule, precedes

# ------------------------ Parameters ------------------------
p = 4                       # @param GPUs / stages
m = p-1                     # @param microbatches per batch (current batch)
T_comm = 0.0                # @param inter-stage comm time
M_B, M_W = 25, 10           # @param memory deltas (GB)
gpu_mem_limit = (p-1) * M_B # @param GPU memory limit (GB)
extra_teach_f = p-1         # @param number of microbatches of the "next iteration" teacher forwards


T = {}
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        T[(stage, mb, 'F_S')] = 1
        T[(stage, mb, 'B')]   = 1
        T[(stage, mb, 'W')]   = 1

# Next iteration: only teacher forwards (no B/W for next iteration in this model)
for stage in range(1, p + 1):
    for mb in range(m + 1, m + extra_teach_f + 1):
        T[(stage, mb, 'F_T')] = 1

def has(stage, mb, op):
    return (stage, mb, op) in T

mdl = pulp.LpProblem("ZB_ILP_student_current_teacher_next", pulp.LpMinimize)
S = {k: pulp.LpVariable(f"S_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
E = {k: pulp.LpVariable(f"E_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
Z = pulp.LpVariable("Z", lowBound=0)

for stage in range(1, p + 1):
    W0 = pulp.LpVariable(f"earliest_FS_stage{stage}", lowBound=0)
    mdl += W0 <= S[(stage, 1, 'F_S')]
    mdl += W0 <= S[(stage, m+1, 'F_T')]

    W1 = pulp.LpVariable(f"last_end_stage{stage}", lowBound=0)
    mdl += W1 >= E[(stage, m, 'W')]
    if has(stage, m + extra_teach_f, 'F_T'):
        mdl += W1 >= E[(stage, m + extra_teach_f, 'F_T')]

    mdl += Z >= W1 - W0

mdl += Z

for k, dur in T.items():
    mdl += E[k] == S[k] + dur

for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        mdl += S[(stage, mb, 'B')] >= E[(stage, mb, 'F_S')] + T_comm
        mdl += S[(stage, mb, 'W')] >= E[(stage, mb, 'B')] + T_comm

for stage in range(1, p + 1):
    for j in range(m + 2, m + extra_teach_f + 1):
        if has(stage, j, 'F_T') and has(stage, j - 1, 'F_T'):
            mdl += S[(stage, j, 'F_T')] >= E[(stage, j - 1, 'F_T')] + T_comm

    for op in ['F_S', 'B', 'W']:
        for j in range(2, m + 1):
            if has(stage, j, op) and has(stage, j - 1, op):
                mdl += S[(stage, j, op)] >= E[(stage, j - 1, op)] + T_comm

for stage in range(2, p + 1):
    for mb in range(1, m + 1):
        mdl += S[(stage, mb, 'F_S')] >= E[(stage - 1, mb, 'F_S')] + T_comm
    for mb in range(m + 1, m + extra_teach_f + 1):
        if has(stage - 1, mb, 'F_T') and has(stage, mb, 'F_T'):
            mdl += S[(stage, mb, 'F_T')] >= E[(stage - 1, mb, 'F_T')] + T_comm

for stage in range(1, p):
    for mb in range(1, m + 1):
        mdl += S[(stage, mb, 'B')] >= E[(stage + 1, mb, 'B')] + T_comm

M = 1e6
y = {}


for stage in range(1, p + 1):
    tasks_on_stage = [task for task in T if task[0] == stage]
    for i, a in enumerate(tasks_on_stage):
        for j, b in enumerate(tasks_on_stage):
            if i >= j:
                continue
            if (a, b) not in y and (b, a) not in y:
                y[(a, b)] = pulp.LpVariable(f"y_{a}_{b}", lowBound=0, upBound=1, cat="Binary")
            mdl += S[b] >= E[a] - M * (1 - precedes(a, b, y))
            mdl += S[a] >= E[b] - M * (1 - precedes(b, a, y))

delta_mem = {'F_S': M_B, 'F_T': 0, 'B': M_W - M_B, 'W': -M_W}

for stage in range(1, p + 1):
    tasks_stage = [t for t in T if t[0] == stage]
    for b in tasks_stage:
        mdl += pulp.lpSum(delta_mem[a[2]] * precedes(a, b, y) for a in tasks_stage) <= gpu_mem_limit



mdl.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*10))
print("Status:", pulp.LpStatus[mdl.status])
print("Objective (Z):", pulp.value(Z))

schedule = defaultdict(list)
for task in sorted(T):
    s = float(pulp.value(S[task]))
    e = float(pulp.value(E[task]))
    schedule[task[0]].append((s, e, task[1], task[2]))


plot_memory_and_schedule(schedule, T, delta_mem, p, m, filename_prefix="zbts_p-1_ilp_intermediate")