import pulp
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

p = 8
m = 2*p + 4
M = 20
ops = ['F_S', 'F_T', 'B', 'W']
M_B, M_W = 25, 10       # Memory usage for B and W operations in GB
M_BBatch, M_WBatch = 50, 20
T_total = 10

# Memory scaling per microbatch
mem_scale = m // 2
delta_mem = {
    'F_S': M_BBatch / m,
    'F_T': 0,
    'B': (M_WBatch / m) - M_BBatch / m,
    'W': -M_WBatch / m
}

# Time scaling per operation
T = {}
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        for op in ops:
            T[(stage, mb, op)] = round(T_total / m, 5)


S = {}


for stage in range(1, p + 1):
    print(f"For stage {stage}: {p-stage} idle steps before starting F_T")
    
    for mb in range(1, p-stage + 2):
        S[(stage, mb, 'F_T')] = (2 * (mb - 1) + stage - 1)*T[(stage, mb, 'F_T')]
        S[(stage, mb, 'F_S')] = S[(stage, mb, 'F_T')] + T[(stage, mb, 'F_T')]
    
    
# repeat steady phase pattern [F_T, F_S, B] for remaining microbatches
for stage in range(1, p + 1):
    for mb in range(p - stage + 2, m + 1):
        # print(f"Scheduling stage {stage} mb {mb}")
        if mb - 1 != 0 :
            S[(stage, mb, 'F_T')] = S[(stage, mb - 1, 'F_T')] + 3*T[(stage, mb - 1, 'F_T')]
            S[(stage, mb, 'F_S')] = S[(stage, mb - 1, 'F_S')] + 3*T[(stage, mb - 1, 'F_S')]
            # S[(stage, mb, 'B')] = S[(stage, mb - 1, 'B')] + 3 
        else :
            S[(stage, mb, 'F_T')] = (p - 1)*T[(stage, mb, 'F_T')]
            S[(stage, mb, 'F_S')] = S[(stage, mb, 'F_T')] + T[(stage, mb, 'F_T')]


# perform the B passes
S[(p, 1, 'B')] = S[(p, 1, 'F_S')] + T[(p, 1, 'F_S')]
for stage in range(p-1, 0, -1):
    print(f"For stage {stage}")
    S[(stage, 1, 'B')] = S[(stage + 1 , 1, 'B')] + T[(stage + 1, 1, 'B')]

for stage in range(1, p + 1):
    for mb in range(2, m + 1):
        S[(stage, mb, 'B')] = S[(stage, mb - 1, 'B')] + 3*T[(stage, mb - 1, 'B')]
        


# Schedule W passes after B passes, shifting if needed
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        # Schedule W after B for this microbatch, avoiding overlaps
        w_start = S[(stage, m-(p-stage), 'B')] + T[(stage, m-(p-stage), 'B')]
        
        print(f"{w_start=}")
        print(f"{[S.get((stage, mb2, op), -1)
            for mb2 in range(1, m + 1)
            for op in ops]}")
        
         # Shift right until no overlap with any existing op in this stage
        while any(
            # (S.get((stage, mb2, op), -10) <= w_start < S.get((stage, mb2, op), -10) + T[(stage, mb2, op)])
            (abs(S.get((stage, mb2, op), -1) - w_start) < 1e-6)
            for mb2 in range(1, m + 1)
            for op in ops if (stage, mb2, op) in S
        ):
            w_start += T[(p, 1, 'F_T')]  # shift by forward chunk duration

        # Assign final start time for W
        S[(stage, mb, 'W')] = w_start



schedule = defaultdict(list)
for task in sorted(T.keys()):
    # task = (stage, mb, op)
    if task in S:
        s = float(S[task])
        e = float(S[task] + T[task])
    else:
        s = float('inf')
        e = float('inf')
    schedule[task[0]].append((s, e, task[1], task[2]))

print(schedule[2])

# Plot schedule (Teacher-Student)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

# --- Top: Gantt chart ---
op_colors = {'F_S': 'royalblue', 'F_T': 'orange', 'B': 'crimson', 'W': 'forestgreen'}
for stage in range(1, p + 1):
    for s, e, mb, op in sorted(schedule[stage]):
        ax1.barh(stage, e - s, left=s, height=0.6,
                 color=op_colors[op], edgecolor='black')
        ax1.text(s + (e - s) / 2, stage, f"{op}{mb}",
                 va='center', ha='center', fontsize=12, color='white')

ax1.set_ylabel("GPU")
ax1.set_yticks(range(1, p + 1))
ax1.set_ylim(0.5, p + 0.5)
ax1.set_title("GPU Operation Schedule (F_S=Forward Student, F_T=Forward Teacher, B=Backward, W=Weight Update)")
ax1.grid(True, linestyle='--', alpha=0.4)
handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
ax1.legend(handles=handles, title='Operations', loc='upper right')

# --- Bottom: Memory usage ---
time_points = range(M + 1)
time_points = range(m*3 + 1)
for stage in range(1, p + 1):
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
plt.savefig("predef_zbts_2p_hand.png")
