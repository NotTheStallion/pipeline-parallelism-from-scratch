import pulp
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

p = 8
m = 2*(p-1)
num_micorbatches = p-1
M = 20
ops = ['F_S', 'F_T', 'B', 'W']
M_B, M_W = 25, 10       # @param Memory usage for B and W operations in GB
M_BBatch, M_WBatch = 50, 20
T_total = 10
alpha = 1.3
delta_mem = {'F_S': M_BBatch/(num_micorbatches//2) + M_WBatch/(num_micorbatches//2), 'F_T': 0, 'B': (- M_BBatch/(num_micorbatches//2) - M_WBatch/(num_micorbatches//2)) /2, 'W': 0}

print(f"Time per microbatch: {T_total/num_micorbatches}, alpha: {alpha}")
print(f"Memory per microbatch: F_S {delta_mem['F_S']}, B {delta_mem['B']}, W {delta_mem['W']}")

T = {}
for stage in range(1, p + 1):
    for mb in range(1, m + 1):
        T[(stage, mb, 'F_S')] = T_total / num_micorbatches
        T[(stage, mb, 'F_T')] = alpha * T[(stage, mb, 'F_S')]
        T[(stage, mb, 'B')] = T_total / num_micorbatches
        T[(stage, mb, 'W')] = T_total / num_micorbatches

S = {}


# Last stage
S[(p, 1, 'F_T')] = (p-1)*T[(p, 1, 'F_T')]
for mb in range(2, (m//2) + 1):
    S[(p, mb, 'F_T')] = S[(p, mb-1, 'F_T')] + T[(p, mb-1, 'F_T')]

S[(p, 1, 'F_S')] = S[(p, m//2, 'F_T')] + T[(p, m//2, 'F_T')]
for mb in range(2, m//2 + 1):
    S[(p, mb, 'F_S')] = S[(p, mb-1, 'F_S')] + T[(p, mb-1, 'F_S')]

S[(p, 1, 'B')] = S[(p, m//2, 'F_S')] + T[(p, m//2, 'F_S')]
for mb in range(2, m + 1):
    S[(p, mb, 'B')] = S[(p, mb-1, 'B')] + T[(p, mb-1, 'B')]

S[(p, (m//2)+1, 'F_T')] = S[(p, m, 'B')] + T[(p, m, 'B')]
for mb in range((m//2)+2, m + 1):
    S[(p, mb, 'F_T')] = S[(p, mb-1, 'F_T')] + T[(p, mb-1, 'F_T')]



# Stage p-1
stage = p-1

for stage in range(p-1, 1, -1):

    S[(stage, 1, 'F_T')] = (stage-1)*T[(stage, 1, 'F_T')]
    for mb in range(2, m//2 + 1):
        S[(stage, mb, 'F_T')] = S[(stage, mb-1, 'F_T')] + T[(stage, mb-1, 'F_T')]

    S[(stage, 1, 'F_S')] = S[(stage, m//2, 'F_T')] + T[(stage, m//2, 'F_T')]
    for mb in range(2, m//2 + 1):
        S[(stage, mb, 'F_S')] = S[(stage, mb-1, 'F_S')] + T[(stage, mb-1, 'F_S')]

    S[(stage, 1, 'B')] = S[(stage+1, 1, 'B')] + T[(stage+1, 1, 'B')]
    for mb in range(2, m + 1):
        S[(stage, mb, 'B')] = S[(stage, mb-1, 'B')] + T[(stage, mb-1, 'B')]



# First stage
S[(1, 1, 'F_T')] = 0
for mb in range(2, m//2 + 1):
    S[(1, mb, 'F_T')] = S[(1, mb-1, 'F_T')] + T[(1, mb-1, 'F_T')]

S[(1, 1, 'F_S')] = S[(1, m//2, 'F_T')] + T[(1, m//2, 'F_T')]
for mb in range(2, m//2 + 1):
    S[(1, mb, 'F_S')] = S[(1, mb-1, 'F_S')] + T[(1, mb-1, 'F_S')]

S[(1, 1, 'B')] = S[(2, 1,'B')] + T[(2, 1, 'B')]
for mb in range(2, m + 1):
    S[(1, mb, 'B')] = S[(1, mb-1, 'B')] + T[(1, mb-1, 'B')]

S[(1, (m//2)+1, 'F_T')] = S[(1, m//2, 'F_S')] + T[(1, m//2, 'F_S')]
for mb in range((m//2)+2, m + 1):
    S[(1, mb, 'F_T')] = S[(1, mb-1, 'F_T')] + T[(1, mb-1, 'F_T')]


for stage in range(2, p+1):
    # first F_T in the second half
    S[(stage, m//2+1, 'F_T')] = S[(stage-1, m//2+1, 'F_T')] + T[(stage, m//2, 'F_T')]

    if stage == p:
        S[(stage, m//2+1, 'F_T')] = S[(stage, m, 'B')] + T[(stage, m, 'B')]

    for mb in range((m//2)+2, m + 1):
        # candidate start
        S[(stage, mb, 'F_T')] = S[(stage, mb-1, 'F_T')] + T[(stage, mb-1, 'F_T')]

        # shift until no overlap with any backward at this stage
        while any(
            (S[(stage, mb, 'F_T')] < S[(stage, mb2, 'B')] + T[(stage, mb2, 'B')]) and
            (S[(stage, mb, 'F_T')] + T[(stage, mb, 'F_T')] > S[(stage, mb2, 'B')])
            for mb2 in range(1, m+1) if (stage, mb2, 'B') in S
        ):
            S[(stage, mb, 'F_T')] = S[(stage, m, 'B')] + T[(stage, m, 'B')]
        S[(stage, mb, 'F_T')] = max(S[(stage, mb, 'F_T')], S[(stage-1, mb, 'F_T')] + T[(stage, mb-1, 'F_T')])



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

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Figure 1: Schedule + Memory ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={'height_ratios': [2, 1]}
)

# --- Top: Gantt chart ---
op_colors = {
    'F_S': 'royalblue',
    'F_T': 'orange',
    'B': 'crimson',
    'W': 'forestgreen'
}

for stage in range(1, p + 1):
    for s, e, mb, op in sorted(schedule[stage]):
        ax1.barh(
            y=stage,
            width=e - s,
            left=s,
            height=0.6,
            color=op_colors[op],
            edgecolor='black'
        )
        # Add operation + microbatch text inside the bar
        ax1.text(
            x=s + (e - s) / 2,
            y=stage,
            s=f"{op}{mb}",
            va='center',
            ha='center',
            fontsize=10,
            color='white',
            fontweight='bold'
        )

ax1.set_ylabel("GPU")
ax1.set_xlabel("Time")   # show x values
ax1.set_yticks(range(1, p + 1))
ax1.set_ylim(0.5, p + 0.5)
ax1.set_title("GPU Operation Schedule\n(F_S=Forward Student, F_T=Forward Teacher, B=Backward, W=Weight Update)")
ax1.grid(True, linestyle='--', alpha=0.4)
handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
ax1.legend(handles=handles, title='Operations', loc='upper right')

# --- Bottom: Memory usage ---
time_points = range(m * 4 + 10)

for stage in range(1, p + 1):
    events = [(s, delta_mem[op]) for s, _, _, op in sorted(schedule[stage], key=lambda x: x[0])]

    mem_timeline = [0]
    cur_mem = 0

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
plt.savefig("predef_tspipe_hand.png")
plt.show()


# --- Figure 2: Only Schedule (Gantt chart) ---
fig2, ax_sched = plt.subplots(figsize=(14, 4))

# Find makespan and per-op total time
stage_makespans = []
for stage in range(1, p + 1):
    starts = [s for s, _, _, op in schedule[stage] if op == 'F_S' and s != float('inf')]
    ends = [e for _, e, _, _ in schedule[stage] if e != float('inf')]
    if starts and ends:
        stage_makespans.append(max(ends) - min(starts))
    else:
        stage_makespans.append(0)
makespan = max(stage_makespans)

# first_start = min(
#     s for stage in range(1, p + 1) for s, _, _, op in schedule[stage] if op == 'F_S'
# )
# last_end = max(e for stage in range(1, p + 1) for _, e, _, _ in schedule[stage])

# print(first_start, last_end)

# makespan = last_end - first_start

op_totals = {op: 0.0 for op in ops}
for stage in range(1, p + 1):
    for s, e, mb, op in sorted(schedule[stage]):
        op_totals[op] += e - s

for stage in range(1, p + 1):
    for s, e, mb, op in sorted(schedule[stage]):
        ax_sched.barh(
            y=stage,
            width=e - s,
            left=s,
            height=0.6,
            color=op_colors[op],
            edgecolor='black'
        )
        # Add operation + microbatch text
        ax_sched.text(
            x=s + (e - s) / 2,
            y=stage,
            s=f"{mb}",
            va='center',
            ha='center',
            fontsize=10,
            color='white',
            fontweight='bold'
        )

ax_sched.set_ylabel("GPU")
ax_sched.set_xlabel("Time")
ax_sched.set_yticks(range(1, p + 1))
ax_sched.set_ylim(0.5, p + 0.5)
ax_sched.set_title("GPU Operation Schedule (Only)")
ax_sched.grid(True, linestyle='--', alpha=0.4)

handles2 = [patches.Patch(color=op_colors[c], label=f"{c}") for c in op_colors]
legend_text = (
    f"Makespan: {makespan:.2f}\n"
    f"T: {T_total:.2f}\nα: {alpha:.2f} "
)
ax_sched.legend(handles=handles2, title=legend_text, loc='upper right')

plt.tight_layout()
plt.savefig("predef_tspipe_hand_schedule_only.png")
plt.show()
