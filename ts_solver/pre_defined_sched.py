import pulp
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

p = 3
m = 3
M = 20
ops = ['F_S', 'F_T', 'B', 'W']
M_B, M_W = 25, 10       # @param Memory usage for B and W operations in GB
delta_mem = {'F_S': M_B, 'F_T': 0, 'B': M_W - M_B, 'W': -M_W}

T = {}
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        for op in ops:
            T[(stage, mb, op)] = 1

S = {}

# last stage
S[(p, 1, 'F_T')] = 2
S[(p, 1, 'F_S')] = 3
S[(p, 1, 'B')] = 4
for mb in range(2, m + 1):
    S[(p, mb, 'F_T')] =  S[(p, mb - 1, 'F_T')] + 3
    S[(p, mb, 'F_S')] =  S[(p, mb - 1, 'F_S')] + 3
    S[(p, mb, 'B')] =  S[(p, mb - 1, 'B')] + 3

S[(p, 1, 'W')] = S[(p, m, 'B')] + 1
for mb in range(2, m + 1):
    S[(p, mb, 'W')] = S[(p, mb - 1, 'W')] + 1


# middle stage
stage = p-1
S[(stage, 1, 'F_T')] = 1
S[(stage, 1, 'F_S')] = 2
S[(stage, m, 'B')] = S[(p, m, 'B')] + 1


S[(stage, 2, 'F_T')] = S[(stage, 1, 'F_T')] + 2
S[(stage, 2, 'F_S')] = S[(stage, 1, 'F_T')] + 3
S[(stage, 1, 'B')] = S[(stage, 1, 'F_T')] + 4


for mb in range(3, m + 1):
    S[(stage, mb, 'F_T')] = S[(stage, mb - 1, 'F_T')] + 3
    S[(stage, mb, 'F_S')] = S[(stage, mb - 1, 'F_T')] + 4
    S[(stage, mb - 1, 'B')] = S[(stage, mb - 1, 'F_T')]  + 5

S[(stage, 1, 'W')] = S[(stage, m-1, 'B')] + 1
S[(stage, 2, 'W')] = S[(stage, 1, 'W')] + 1

S[(stage, 3, 'W')] = S[(stage, m, 'B')] + 1
for mb in range(4, m + 1):
    S[(stage, mb, 'W')] = S[(stage, mb - 1, 'W')] + 1



# first stage
stage = 1
S[(stage, 1, 'F_T')] = 0
S[(stage, 1, 'F_S')] = 1
for mb in range(2, m + 1):
    S[(stage, mb, 'F_T')] = S[(stage, mb - 1, 'F_T')] + 2
    S[(stage, mb, 'F_S')] = S[(stage, mb - 1, 'F_T')] + 3

S[(stage, 1, 'B')] = S[(stage, 1, 'F_T')] + 6
S[(stage, 1, 'W')] = S[(stage, 1, 'B')] + 1
for mb in range(2, m + 1):
    S[(stage, mb, 'B')] = S[(stage, mb - 1, 'B')] + 3
    S[(stage, mb, 'W')] = S[(stage, mb, 'B')] + 1


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
                 va='center', ha='center', fontsize=8, color='white')

ax1.set_ylabel("GPU")
ax1.set_yticks(range(1, p + 1))
ax1.set_ylim(0.5, p + 0.5)
ax1.set_title("GPU Operation Schedule (F_S=Forward Student, F_T=Forward Teacher, B=Backward, W=Weight Update)")
ax1.grid(True, linestyle='--', alpha=0.4)
handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
ax1.legend(handles=handles, title='Operations', loc='upper right')

# --- Bottom: Memory usage ---
time_points = range(M + 1)
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
plt.savefig("predef_ts_zb.png")
