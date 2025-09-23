import pulp
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

p = 8
m = 2*(p-1)
M = 20
ops = ['F_S', 'F_T', 'B', 'W']
M_B, M_W = 25, 10       # @param Memory usage for B and W operations in GB
M_BBatch, M_WBatch = 50, 20
delta_mem = {'F_S': M_BBatch/(m//2) + M_WBatch/(m//2), 'F_T': 0, 'B': (- M_BBatch/(m//2) - M_WBatch/(m//2)) /2, 'W': -M_W}

T = {}
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        for op in ops:
            T[(stage, mb, op)] = 1

S = {}


# Last stage
S[(p, 1, 'F_T')] = p-1
for mb in range(2, (m//2) + 1):
    S[(p, mb, 'F_T')] = S[(p, mb-1, 'F_T')] + 1

S[(p, 1, 'F_S')] = S[(p, m//2, 'F_T')] + 1
for mb in range(2, m//2 + 1):
    S[(p, mb, 'F_S')] = S[(p, mb-1, 'F_S')] + 1

S[(p, 1, 'B')] = S[(p, m//2, 'F_S')] + 1
for mb in range(2, m + 1):
    S[(p, mb, 'B')] = S[(p, mb-1, 'B')] + 1

S[(p, (m//2)+1, 'F_T')] = S[(p, m, 'B')] + 1
for mb in range((m//2)+2, m + 1):
    S[(p, mb, 'F_T')] = S[(p, mb-1, 'F_T')] + 1



# Stage p-1
stage = p-1

for stage in range(p-1, 1, -1):

    S[(stage, 1, 'F_T')] = stage-1
    for mb in range(2, m//2 + 1):
        S[(stage, mb, 'F_T')] = S[(stage, mb-1, 'F_T')] + 1

    S[(stage, 1, 'F_S')] = S[(stage, m//2, 'F_T')] + 1
    for mb in range(2, m//2 + 1):
        S[(stage, mb, 'F_S')] = S[(stage, mb-1, 'F_S')] + 1

    S[(stage, 1, 'B')] = S[(stage+1, 1, 'B')] + 1
    for mb in range(2, m + 1):
        S[(stage, mb, 'B')] = S[(stage, mb-1, 'B')] + 1

    S[(stage, (m//2)+1, 'F_T')] = S[(stage, m//2, 'F_S')] + 1
    for mb in range((m//2)+2, m + 1):
        S[(stage, mb, 'F_T')] = S[(stage, mb-1, 'F_T')] + 1
        if S[(stage, mb, 'F_T')] == S[(stage, 1, 'B')]:
            S[(stage, mb, 'F_T')] += 2*(p-1)



# First stage
S[(1, 1, 'F_T')] = 0
for mb in range(2, m//2 + 1):
    S[(1, mb, 'F_T')] = S[(1, mb-1, 'F_T')] + 1

S[(1, 1, 'F_S')] = S[(1, m//2, 'F_T')] + 1
for mb in range(2, m//2 + 1):
    S[(1, mb, 'F_S')] = S[(1, mb-1, 'F_S')] + 1

S[(1, 1, 'B')] = S[(1, m//2, 'F_S')] + 1 + (p-1)*2
for mb in range(2, m + 1):
    S[(1, mb, 'B')] = S[(1, mb-1, 'B')] + 1

S[(1, (m//2)+1, 'F_T')] = S[(1, m//2, 'F_S')] + 1
for mb in range((m//2)+2, m + 1):
    S[(1, mb, 'F_T')] = S[(1, mb-1, 'F_T')] + 1

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
        if op in []:
            ax1.text(s + (e - s) / 2, stage, f"{op}{((mb-1)//2)+1}",
                va='center', ha='center', fontsize=12, color='white')  # Increased fontsize
        else:
            ax1.text(s + (e - s) / 2, stage, f"{op}{mb}",
                    va='center', ha='center', fontsize=12, color='white')  # Increased fontsize

ax1.set_ylabel("GPU", fontsize=12)  # Increased fontsize
ax1.set_yticks(range(1, p + 1))
ax1.set_ylim(0.5, p + 0.5)
ax1.set_title("GPU Operation Schedule (F_S=Forward Student, F_T=Forward Teacher, B=Backward, W=Weight Update)", fontsize=14)  # Increased fontsize
ax1.grid(True, linestyle='--', alpha=0.4)
handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
ax1.legend(handles=handles, title='Operations', loc='upper right', fontsize=10, title_fontsize=12)  # Increased fontsize

# --- Bottom: Memory usage ---
time_points = range(50 + 1)
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

ax2.set_xlabel("Time", fontsize=12)  # Increased fontsize
ax2.set_ylabel("Memory (GB)", fontsize=12)  # Increased fontsize
ax2.set_title("Per-GPU Memory Usage Over Time", fontsize=14)  # Increased fontsize
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(fontsize=13)  # Increased fontsize

plt.tight_layout()
plt.savefig("res_tspipe.png")