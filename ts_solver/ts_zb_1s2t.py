from collections import defaultdict
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------ Parameters ------------------------
p = 3                    # GPUs / stages
m = 2                    # microbatches per batch
T_comm = 0.0             # inter-stage comm time
M_B, M_W = 25, 10        # memory deltas (GB)
gpu_mem_limit = p*M_B  # GPU memory limit (GB)

times = 2

# ------------------------ Tasks -----------------------------
# T[(stage, mb, op)] = duration
T = {}
for stage in range(1, p+1):
    for mb in range(1, m+1):
        T[(stage, mb, 'F_S')] = 1
        T[(stage, mb, 'F_T')] = 1
        T[(stage, mb, 'B')]   = 1
        T[(stage, mb, 'W')]   = 1

# Next batch: teacher forwards only
for stage in range(1, p+1):
    for mb in range(m+1, times+m+1):
        T[(stage, mb, 'F_T')] = 1

def has(stage, mb, op):
    return (stage, mb, op) in T

# ------------------------ Model -----------------------------
mdl = pulp.LpProblem("ZB_ILP_fixed_ordered_next_batch_teacher_only", pulp.LpMinimize)
S = {k: pulp.LpVariable(f"S_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
E = {k: pulp.LpVariable(f"E_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
Z = pulp.LpVariable("Z", lowBound=0)

# Objective: minimize (per-stage last W of first batch) – (earliest first-batch forward start)
for stage in range(1, p+1):
    # earliest of F_S1 and F_T1 (first-batch only)
    W0 = pulp.LpVariable(f"W_{stage}", lowBound=0)
    mdl += W0 <= S[(stage, 1, 'F_S')]
    mdl += W0 <= S[(stage, 3, 'F_T')]
    
    
    # max W_m and F_T_2m
    W1 = pulp.LpVariable(f"W1_{stage}", lowBound=0)
    mdl += W1 >= E[(stage, m, 'W')]
    mdl += W1 >= E[(stage, times+m, 'F_T')]
    # last W of first batch is at mb = m
    mdl += Z >= W1 - W0
mdl += Z

# ------------------------ Temporal relations ----------------
# Start-end
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
    for j in range(2, times+m+1):
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
    for mb in range(1, times+m+1):
        if has(stage-1, mb, 'F_T') and has(stage, mb, 'F_T'):
            mdl += S[(stage, mb, 'F_T')] >= E[(stage-1, mb, 'F_T')] + T_comm

# Backward dependency (first batch only)
for stage in range(1, p):
    for mb in range(1, m+1):
        mdl += S[(stage, mb, 'B')] >= E[(stage+1, mb, 'B')] + T_comm

# ------------------------ Same-stage non-overlap (big-M) ----
# Disjunctive resource on each stage for all tasks present on that stage
M = 1e5
y = {}  # precedence binaries keyed by (a,b) for a!=b on same stage

def precedes(a, b):
    if a == b:
        return 1
    if (a, b) in y:
        return y[(a, b)]
    elif (b, a) in y:
        return 1 - y[(b, a)]
    else:
        raise RuntimeError(f"Missing precedence variable for {a} and {b}")

for stage in range(1, p+1):
    tasks_on_stage = [task for task in T if task[0] == stage]
    for a in tasks_on_stage:
        for b in tasks_on_stage:
            if a == b:
                continue
            if (a, b) not in y and (b, a) not in y:
                y[(a, b)] = pulp.LpVariable(f"y_{a}_{b}", lowBound=0, upBound=1, cat="Binary")
                # Mutually exclusive ordering on end-times (works since E = S + dur)
                mdl += E[a] >= E[b] + T[a] - M * precedes(a, b)
                mdl += E[b] >= E[a] + T[b] - M * precedes(b, a)

# ------------------------ Memory capacity -------------------
delta_mem = {'F_S': M_B, 'F_T':0, 'B': M_W - M_B, 'W': -M_W}

for stage in range(1, p+1):
    tasks_stage = [t for t in T if t[0] == stage]
    for b in tasks_stage:
        # capacity at every completion instant E[b]
        mdl += pulp.lpSum(delta_mem[a[2]] * precedes(a, b) for a in tasks_stage) <= gpu_mem_limit

# ------------------------ Solve & Report --------------------
mdl.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*60*4))
print("Status:", pulp.LpStatus[mdl.status])
print("Objective (Z):", pulp.value(Z))

schedule = defaultdict(list)
for task in sorted(T):
    s = float(pulp.value(S[task]))
    e = float(pulp.value(E[task]))
    schedule[task[0]].append((s, e, task[1], task[2]))

print("Stage 1 sample:", sorted(schedule[1])[:10])

# ------------------------ Plot (Gantt + Memory) -------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})

op_colors = {'F_S': 'royalblue', 'F_T': 'orange', 'B': 'crimson', 'W': 'forestgreen'}

# Gantt
for stage in range(1, p+1):
    for s, e, mb, op in sorted(schedule[stage]):
        ax1.barh(stage, e - s, left=s, height=0.6,
                 color=op_colors[op], edgecolor='black')
        ax1.text(s + (e - s)/2, stage, f"{op}{mb}", va='center', ha='center',
                 fontsize=12, color='white')  # Increased fontsize
ax1.set_ylabel("GPU", fontsize=12)  # Increased fontsize
ax1.set_yticks(range(1, p+1))
ax1.set_ylim(0.5, p + 0.5)
ax1.set_title("GPU Operation Schedule (F=Forward, B=Backward, W=Weight Update)", fontsize=14)  # Increased fontsize
ax1.grid(True, linestyle='--', alpha=0.4)
handles = [patches.Patch(color=op_colors[c]) for c in op_colors]
labels = list(op_colors.keys())
ax1.legend(handles, labels, title='Operations', loc='upper right', fontsize=10, title_fontsize=12)  # Increased fontsize

# Memory (event-based from starts; for visualization only)
def horizon_upper_bound():
    dmax = max(T.values())
    return int(3 * m * dmax + 2 * (p - 1) * (dmax + T_comm) + 5)

time_points = range(horizon_upper_bound() + 1)
time_points = range(15)  # Adjusted for better visualization
for stage in range(1, p+1):
    events = sorted((s, op) for s, e, mb, op in schedule[stage])
    mem_timeline = []
    cur_mem = 0
    idx = 0
    for t in time_points:
        while idx < len(events) and events[idx][0] < t:  # Adjusted to "< t" for alignment
            cur_mem += delta_mem[events[idx][1]]
            idx += 1
        mem_timeline.append(cur_mem)
    ax2.plot(time_points, mem_timeline, label=f"GPU{stage}")

ax2.set_xlabel("Time", fontsize=12)  # Increased fontsize
ax2.set_ylabel("Memory (GB)", fontsize=12)  # Increased fontsize
ax2.set_title("Per-GPU Memory Usage Over Time", fontsize=14)  # Increased fontsize
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(fontsize=13)  # Increased fontsize
plt.tight_layout()
plt.savefig("ts_zb_3.png")
