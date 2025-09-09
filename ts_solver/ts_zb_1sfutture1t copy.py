from collections import defaultdict
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------ Parameters ------------------------
p = 3                    # GPUs / stages
m = 2                    # microbatches per batch (current batch)
T_comm = 0.0             # inter-stage comm time
M_B, M_W = 25, 10        # memory deltas (GB)
gpu_mem_limit = (p-1) * M_B - 1  # GPU memory limit (GB)

times = 2  # number of microbatches of the "next iteration" we model for teacher forwards

# ------------------------ Tasks -----------------------------
# T[(stage, mb, op)] = duration
T = {}
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        # Current batch: only student forwards, backward, weight update
        T[(stage, mb, 'F_S')] = 1
        T[(stage, mb, 'B')]   = 1
        T[(stage, mb, 'W')]   = 1

# Next iteration: only teacher forwards (no B/W for next iteration in this model)
for stage in range(1, p + 1):
    for mb in range(m + 1, m + times + 1):
        T[(stage, mb, 'F_T')] = 1

def has(stage, mb, op):
    return (stage, mb, op) in T

# ------------------------ Model -----------------------------
mdl = pulp.LpProblem("ZB_ILP_student_current_teacher_next", pulp.LpMinimize)
S = {k: pulp.LpVariable(f"S_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
E = {k: pulp.LpVariable(f"E_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
Z = pulp.LpVariable("Z", lowBound=0)

# Objective: minimize (per-stage last W of first batch and last teacher forward of next iter)
# minus (earliest student-forward start of first microbatch) aggregated via Z
for stage in range(1, p + 1):
    # earliest student-forward start for mb=1 on this stage (current batch)
    W0 = pulp.LpVariable(f"earliest_FS_stage{stage}", lowBound=0)
    # enforce W0 <= start of the student forward of mb=1 (exists by construction)
    mdl += W0 <= S[(stage, 1, 'F_S')]
    mdl += W0 <= S[(stage, m+1, 'F_T')]  # also consider the teacher forward of mb=1 of next iter

    # W1: max between last W of current batch (mb = m) and last teacher forward of modeled next iter (mb = m+times)
    W1 = pulp.LpVariable(f"last_end_stage{stage}", lowBound=0)
    mdl += W1 >= E[(stage, m, 'W')]              # last W of the current batch
    # last teacher forward present is at mb = m + times
    if has(stage, m + times, 'F_T'):
        mdl += W1 >= E[(stage, m + times, 'F_T')]

    mdl += Z >= W1 - W0

mdl += Z  # objective

# ------------------------ Temporal relations ----------------
# Start-end
for k, dur in T.items():
    mdl += E[k] == S[k] + dur

# Intra-microbatch chaining (current batch only, student forwards)
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        mdl += S[(stage, mb, 'B')] >= E[(stage, mb, 'F_S')] + T_comm
        mdl += S[(stage, mb, 'W')] >= E[(stage, mb, 'B')] + T_comm

# Microbatch order per op on each stage
for stage in range(1, p + 1):
    # F_T must be ordered across its domain (next iteration microbatches)
    for j in range(m + 2, m + times + 1):
        if has(stage, j, 'F_T') and has(stage, j - 1, 'F_T'):
            mdl += S[(stage, j, 'F_T')] >= E[(stage, j - 1, 'F_T')] + T_comm

    # F_S, B, W are only for current batch (1..m) and must be ordered
    for op in ['F_S', 'B', 'W']:
        for j in range(2, m + 1):
            if has(stage, j, op) and has(stage, j - 1, op):
                mdl += S[(stage, j, op)] >= E[(stage, j - 1, op)] + T_comm

# Inter-stage forward deps
for stage in range(2, p + 1):
    # F_S flows only for current batch
    for mb in range(1, m + 1):
        mdl += S[(stage, mb, 'F_S')] >= E[(stage - 1, mb, 'F_S')] + T_comm
    # F_T flows only for next-iteration microbatches (guarded)
    for mb in range(m + 1, m + times + 1):
        if has(stage - 1, mb, 'F_T') and has(stage, mb, 'F_T'):
            mdl += S[(stage, mb, 'F_T')] >= E[(stage - 1, mb, 'F_T')] + T_comm

# Backward dependency (current batch only)
for stage in range(1, p):
    for mb in range(1, m + 1):
        mdl += S[(stage, mb, 'B')] >= E[(stage + 1, mb, 'B')] + T_comm

# ------------------------ Same-stage non-overlap (big-M) ----
# Disjunctive resource on each stage for all tasks present on that stage
M = 1e6
y = {}  # precedence binaries keyed by (a,b) for a!=b on same stage

def precedes(a, b):
    # returns expression (1 if a before b, else binary var or 1 - var)
    if a == b:
        return 1
    if (a, b) in y:
        return y[(a, b)]
    elif (b, a) in y:
        return 1 - y[(b, a)]
    else:
        raise RuntimeError(f"Missing precedence variable for {a} and {b}")

for stage in range(1, p + 1):
    tasks_on_stage = [task for task in T if task[0] == stage]
    for i, a in enumerate(tasks_on_stage):
        for j, b in enumerate(tasks_on_stage):
            if i >= j:
                continue
            # create a single binary for the unordered pair (a,b)
            if (a, b) not in y and (b, a) not in y:
                y[(a, b)] = pulp.LpVariable(f"y_{a}_{b}", lowBound=0, upBound=1, cat="Binary")
            # Use standard big-M disjunction on START times:
            # if y[a,b] == 1 => a before b (S[b] >= E[a])
            # else => b before a (S[a] >= E[b])
            mdl += S[b] >= E[a] - M * (1 - precedes(a, b))
            mdl += S[a] >= E[b] - M * (1 - precedes(b, a))

# ------------------------ Memory capacity -------------------
delta_mem = {'F_S': M_B, 'F_T': 0, 'B': M_W - M_B, 'W': -M_W}

for stage in range(1, p + 1):
    tasks_stage = [t for t in T if t[0] == stage]
    for b in tasks_stage:
        # enforce capacity at completion instant E[b]:
        # Sum of memory deltas of tasks that finish before (or at) E[b] <= gpu_mem_limit.
        # We approximate "finished before E[b]" by the precedence binaries: precedes(a,b) == 1 => a before b
        mdl += pulp.lpSum(delta_mem[a[2]] * precedes(a, b) for a in tasks_stage) <= gpu_mem_limit

# ------------------------ Solve & Report --------------------
mdl.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60*10))
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
for stage in range(1, p + 1):
    for s, e, mb, op in sorted(schedule[stage]):
        ax1.barh(stage, e - s, left=s, height=0.6,
                 color=op_colors[op], edgecolor='black')
        ax1.text(s + (e - s) / 2, stage, f"{op}{mb}", va='center', ha='center',
                 fontsize=10, color='white')
ax1.set_ylabel("GPU")
ax1.set_yticks(range(1, p + 1))
ax1.set_ylim(0.5, p + 0.5)
ax1.set_title("GPU Operation Schedule (F=Forward, B=Backward, W=Weight Update)")
ax1.grid(True, linestyle='--', alpha=0.4)
handles = [patches.Patch(color=op_colors[c]) for c in op_colors]
labels = list(op_colors.keys())
ax1.legend(handles, labels, title='Operations', loc='upper right')

# Memory (event-based from starts; for visualization only)
def horizon_upper_bound():
    dmax = max(T.values())
    return int(3 * m * dmax + 2 * (p - 1) * (dmax + T_comm) + 5)

time_points = range(15)
for stage in range(1, p + 1):
    events = sorted((s, op) for s, e, mb, op in schedule[stage])
    mem_timeline = []
    cur_mem = 0
    idx = 0
    for t in time_points:
        while idx < len(events) and events[idx][0] < t:
            cur_mem += delta_mem[events[idx][1]]
            idx += 1
        mem_timeline.append(cur_mem)
    ax2.plot(time_points, mem_timeline, label=f"GPU{stage}")

ax2.set_xlabel("Time")
ax2.set_ylabel("Memory (GB)")
ax2.set_title("Per-GPU Memory Usage Over Time")
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend()
plt.tight_layout()
plt.savefig("ts_zb_3_modified.png")
