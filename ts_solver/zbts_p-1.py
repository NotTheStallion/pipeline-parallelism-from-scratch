import pulp
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

p = 24
m = p-1
M = 20
ops = ['F_S', 'F_T', 'B', 'W']
M_B, M_W = 25, 10       # @param Memory usage for B and W operations in GB
delta_mem = {'F_S': M_B, 'F_T': 0, 'B': M_W - M_B, 'W': -M_W}

T = {}
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        for op in ops:
            T[(stage, mb, op)] = 1


# additional teacher forwards
for stage in range(1, p + 1):
    for mb in range(m + 1, 2*m + 1):
        T[(stage, mb, 'F_T')] = 1

S = {}

# p-1 warmup teacher forwards

for mb in range(1, m + 1):
    S[(1, mb, 'F_T')] = mb - 1

for stage in range(2, p+1):
    for mb in range(1, m + 1):
        S[(stage, mb, 'F_T')] = S[(stage - 1, mb, 'F_T')] + 1
        
# interleaved student forwards

for stage in range(1, p + 1):
    S[(stage, 1, 'F_S')] = S[(stage, m-1, 'F_T')] + 1 + 1
    for mb in range(2, m + 1):
        S[(stage, mb, 'F_S')] = S[(stage, mb - 1, 'F_S')] + 1 + 1

# interleaved backwards

for mb in range(1, m + 1):
    S[(p, mb, 'B')] = S[(p, mb, 'F_S')] + 1

for stage in range(p - 1, 0, -1):
    for mb in range(1, m + 1):
        S[(stage, mb, 'B')] = S[(stage + 1, mb, 'B')] + 1
        
# Schedule W passes after B passes, shifting if needed
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        # Schedule W after B for this microbatch, avoiding overlaps
        w_start = S[(stage, mb, 'B')] + T[(stage, mb, 'B')]
        
        while any(
            (S.get((stage, mb2, op), -1) == w_start)
            for mb2 in range(1, m + 1)
            for op in ops
        ):
            w_start += 1
        S[(stage, mb, 'W')] = w_start
    

# Fill bubble with next batch teacher forwards
for stage in range(1, p + 1):
    for mb in range(m+1, 2*m + 1):
        # Schedule F_T after F_S of previous microbatch, avoiding overlaps
        print(mb)
        f_start = S[(stage, mb-m, 'F_S')] + T[(stage, mb-m, 'F_S')]
        
        while any(
            (S.get((stage, mb2, op), -1) == f_start)
            for mb2 in range(1, 2*m + 1)
            for op in ops
        ):
            f_start += 1
        S[(stage, mb, 'F_T')] = f_start





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
        # ax1.text(s + (e - s) / 2, stage, f"{op}{mb}",
        #          va='center', ha='center', fontsize=12, color='white')

ax1.set_ylabel("GPU")
ax1.set_yticks(range(1, p + 1))
ax1.set_ylim(0.5, p + 0.5)
ax1.set_title("GPU Operation Schedule (F_S=Forward Student, F_T=Forward Teacher, B=Backward, W=Weight Update)")
ax1.grid(True, linestyle='--', alpha=0.4)
handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
ax1.legend(handles=handles, title='Operations', loc='upper right')

# --- Bottom: Memory usage ---
time_points = range(M + 1)
time_points = range(14 + 1)
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
plt.savefig("predef_zbts_p-1_hand.png")
