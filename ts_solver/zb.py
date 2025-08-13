import itertools
from collections import defaultdict
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches


p = 4                   # @param GPUs
m = 4                   # @param microbatches
T_comm = 0.0            # @param inter-stage communication time
gpu_mem_limit = 100     # @param GPU memory limit in GB
M_B, M_W = 25, 10       # @param Memory usage for B and W operations in GB

T = {}
for stage in range(1, p+1):
    for mb in range(1, m+1):
        T[(stage, mb, 'F')] = 1
        T[(stage, mb, 'B')] = 1
        T[(stage, mb, 'W')] = 1

mdl = pulp.LpProblem("ZB_ILP_fixed_ordered", pulp.LpMinimize)

S = {k: pulp.LpVariable(f"S_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
E = {k: pulp.LpVariable(f"E_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}

# Objective
Z = pulp.LpVariable("Z", lowBound=0)
for stage in range(1, p+1):
    mdl += Z >= E[(stage, m, 'W')] - S[(stage, 1, 'F')]
mdl += Z

# start end relation
for k, dur in T.items():
    mdl += E[k] >= S[k] + dur

# F -> B -> W
for stage in range(1, p+1):
    for mb in range(1, m+1):
        mdl += S[(stage, mb, 'B')] >= E[(stage, mb, 'F')] + T_comm
        mdl += S[(stage, mb, 'W')] >= E[(stage, mb, 'B')] + T_comm

# microbatch order
ops = ['F','B','W']
for stage in range(1, p+1):
    for op in ops:
        for j in range(2, m+1):
            mdl += S[(stage, j, op)] >= E[(stage, j-1, op)] + T_comm

# Forward dependency
for stage in range(2, p+1):
    for mb in range(1, m+1):
        mdl += S[(stage, mb, 'F')] >= E[(stage-1, mb, 'F')] + T_comm

# Backward dependency
for stage in range(1, p):
    for mb in range(1, m+1):
        mdl += S[(stage, mb, 'B')] >= E[(stage+1, mb, 'B')] + T_comm


# @note : No computation overlap (GPT5 solution)
def horizon_upper_bound():
    dmax = max(T.values())
    return int(3 * m * dmax + 2 * (p - 1) * (dmax + T_comm) + 5)


def precedes(a, b):
    if a == b:
        return 1
    
    if (a, b) in y:
        return y[(a, b)]
    elif (b, a) in y:
        return 1 - y[(b, a)]
    else:
        raise ValueError(f"No precedence relation defined for {a} and {b}")
    
    

M = 1e5 # horizon_upper_()bound
y = {}
for stage in range(1, p+1):
    tasks_on_stage = [task for task in T.keys() if task[0] == stage]
    for a in tasks_on_stage:
        for b in tasks_on_stage:
            if (a,b) not in y and (b,a) not in y:
                y[(a, b)] = pulp.LpVariable(f"y_{a}_{b}", lowBound=0, upBound=1, cat="Binary")
                mdl += S[a] >= E[b] - M * precedes(b, a)
                mdl += S[b] >= E[a] - M * precedes(a, b)




# Memory limit constraint
delta_mem = {'F': M_B, 'B': M_W - M_B, 'W': -M_W}

for stage in range(1, p+1):
    tasks_stage = [t for t in T.keys() if t[0] == stage]

    # Enforce capacity at every finish instant E[t]
    for b in tasks_stage:
        # Sum deltas of all tasks u whose E[a] <= E[b]
        mem_prefix_terms = []
        for a in tasks_stage:
            mem_prefix_terms.append(delta_mem[a[2]] * precedes(a, b))

        mdl += pulp.lpSum(mem_prefix_terms) <= gpu_mem_limit





mdl.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*6))
print("Status:", pulp.LpStatus[mdl.status])
print("Objective (Z):", pulp.value(Z))

schedule = defaultdict(list)
for task in sorted(T.keys()):
    # task = (stage, mb, op)
    s = float(pulp.value(S[task]))
    e = float(pulp.value(E[task]))
    schedule[task[0]].append((s, e, task[1], task[2]))

print(schedule[2])




# Plot schedule (GPT5)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})

# --- Top: Gantt chart ---
op_colors = {'F': 'royalblue', 'B': 'crimson', 'W': 'forestgreen'}
for stage in range(1, p+1):
    for s, e, mb, op in sorted(schedule[stage]):
        ax1.barh(stage, e - s, left=s, height=0.6,
                 color=op_colors[op], edgecolor='black')
        ax1.text(s + (e - s) / 2, stage, f"{op}{mb}",
                 va='center', ha='center', fontsize=8, color='white')

ax1.set_ylabel("GPU")
ax1.set_yticks(range(1, p+1))
ax1.set_ylim(0.5, p + 0.5)
ax1.set_title("GPU Operation Schedule (F=Forward, B=Backward, W=Weight Update)")
ax1.grid(True, linestyle='--', alpha=0.4)
handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
ax1.legend(handles=handles, title='Operations', loc='upper right')

# --- Bottom: Memory usage ---
time_points = range(horizon_upper_bound() + 1)
for stage in range(1, p+1):
    events = []
    for s, e, mb, op in sorted(schedule[stage], key=lambda x: x[0]):
        events.append((s, delta_mem[op]))
    mem_timeline = [0]
    cur_mem = 0
    last_t = 0
    for t in time_points:
        while events and events[0][0] <= t:
            _, delta = events.pop(0)
            cur_mem += delta
        mem_timeline.append(cur_mem)
    ax2.plot(time_points, mem_timeline[:-1], label=f"GPU{stage}")


ax2.set_xlabel("Time")
ax2.set_ylabel("Memory (GB)")
ax2.set_title("Per-GPU Memory Usage Over Time")
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend()

plt.tight_layout()
plt.show()

