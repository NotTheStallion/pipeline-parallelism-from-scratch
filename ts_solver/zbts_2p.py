import pulp
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def zbts_2p(p=4, m=8, M_BBatch=50, M_WBatch=20, T_total=10, alpha=1):

    ops = ['F_S', 'F_T', 'B', 'W']
    delta_mem = {
        'F_S': M_BBatch / m,
        'F_T': 0,
        'B': (M_WBatch / m) - M_BBatch / m,
        'W': -M_WBatch / m
    }


    T = {}
    for stage in range(1, p + 1):
        for mb in range(1, m + 1):
            for op in ops:
                if op == 'F_T':
                    T[(stage, mb, op)] = alpha * (T_total / m)
                else:
                    T[(stage, mb, op)] = T_total / m


    S = {}


    for stage in range(1, p + 1):
        print(f"For stage {stage}: {p-stage} idle steps before starting F_T")
        print((mb - 1) * T[(1,1,'F_T')] + (mb - 1) * T[(1,1,'F_S')] + (stage - 1)*T[(1,1,'F_T')])
        for mb in range(1, p-stage + 2):
            S[(stage, mb, 'F_T')] = max((mb - 1) * T[(1,1,'F_T')] + (mb - 1) * T[(1,1,'F_S')] + (stage - 1)*T[(1,1,'F_T')], (mb - 1) * T[(1,1,'F_T')] + (mb - 1) * T[(1,1,'F_S')] + (stage - 1)*T[(1,1,'F_S')])#, S[(stage-1, mb, 'F_T')] + T[(stage-1, mb, 'F_T')] if stage > 1 else 0, S[(stage, mb-1, 'F_S')] + T[(stage, mb-1, 'F_S')] if mb > 1 else 0, S[(stage, mb, 'B')] + T[(stage, mb, 'B')])
            S[(stage, mb, 'F_S')] = max(S[(stage, mb, 'F_T')] + T[(stage, mb, 'F_T')], S[(stage-1, mb, 'F_S')] + T[(stage-1, mb, 'F_S')] if stage > 1 else 0)
        
        
    # repeat steady phase pattern [F_T, F_S, B] for remaining microbatches
    for stage in range(p, 0, -1):
        if stage == p:
            for mb in range(p - stage + 2, m + 1):
                S[(stage, mb, 'F_T')] = S[(stage, mb - 1, 'F_T')] + T[(stage, mb - 1, 'F_T')] + T[(stage, mb - 1, 'F_S')] + T[(stage, mb - 1, 'B')]
                S[(stage, mb, 'F_S')] = S[(stage, mb - 1, 'F_S')] + T[(stage, mb - 1, 'F_T')] + T[(stage, mb - 1, 'F_S')] + T[(stage, mb - 1, 'B')]
        else:
            for mb in range(p - stage + 2, m + 1):
                print(f"Stage {stage}, mb {mb}")
                S[(stage, mb, 'F_T')] = S[(stage + 1, mb-1, 'F_T')] + T[(stage + 1, mb, 'B')]
                S[(stage, mb, 'F_S')] = S[(stage , mb, 'F_T')] + T[(stage, mb, 'F_T')]



    # perform the B passes
    S[(p, 1, 'B')] = S[(p, 1, 'F_S')] + T[(p, 1, 'F_S')]
    for stage in range(p-1, 0, -1):
        print(f"For stage {stage}")
        S[(stage, 1, 'B')] = max(S[(stage + 1 , 1, 'B')] + T[(stage + 1, 1, 'B')], S[(stage, 1+(p-stage), 'F_S')] + T[(stage, 1+(p-stage), 'F_S')])

    for stage in range(1, p + 1):
        for mb in range(2, m + 1):
            S[(stage, mb, 'B')] = S[(stage, mb - 1, 'B')] + T[(stage, mb - 1, 'B')] + T[(stage, mb, 'F_T')] + T[(stage, mb, 'F_S')]
            


    # Schedule W passes after each B pass, ASAP without overlaps
    for stage in range(1, p + 1):
        for mb in range(1, m + 1):
            # Get finish time of B for the same microbatch
            b_finish = S[(stage, mb, 'B')] + T[(stage, mb, 'B')]
            w_start = b_finish

            # Collect all existing ops in this stage
            stage_ops = [
                (S[(stage, mb2, op)], S[(stage, mb2, op)] + T[(stage, mb2, op)])
                for mb2 in range(1, m + 1)
                for op in ops if (stage, mb2, op) in S
            ]

            # Shift W forward until no overlap
            while any(start < w_start + T[(stage, mb, 'W')] and w_start < end for start, end in stage_ops):
                w_start = max(
                    end
                    for start, end in stage_ops
                    if start < w_start + T[(stage, mb, 'W')] and w_start < end
                )

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

    # Find makespan and per-op total time
    stage_makespans = []
    for stage in range(1, p + 1):
        student_forward_starts = [s for s, _, _, _ in schedule[stage]]
        all_ends = [e for _, e, _, _ in schedule[stage]]
        if student_forward_starts and all_ends:
            makespan = max(all_ends) - min(student_forward_starts)
            stage_makespans.append(makespan)
        else:
            stage_makespans.append(0)
    print(f"{stage_makespans=}")
    makespan = max(stage_makespans)

    return S, T, schedule, makespan, delta_mem

if __name__ == "__main__":
    from zbts_p_1 import plot_schedule_and_memory
    
    p = 4
    m = 2*p# + int(alpha)
    M_BBatch, M_WBatch = 50, 20
    T_total = 10
    alpha = 1
    
    S, T, schedule, makespan, delta_mem = zbts_2p(p=p, m=m, M_BBatch=M_BBatch, M_WBatch=M_WBatch, T_total=T_total, alpha=alpha)
    
    plot_schedule_and_memory(S, T, schedule, makespan, delta_mem, p, m, T_total, alpha, filename_prefix="predef_zbts_2p_hand")


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # --- Figure 1: Schedule + Memory ---
# fig, (ax1, ax2) = plt.subplots(
#     2, 1, figsize=(14, 8), sharex=True,
#     gridspec_kw={'height_ratios': [2, 1]}
# )

# # --- Top: Gantt chart ---
# op_colors = {
#     'F_S': 'royalblue',
#     'F_T': 'orange',
#     'B': 'crimson',
#     'W': 'forestgreen'
# }

# for stage in range(1, p + 1):
#     for s, e, mb, op in sorted(schedule[stage]):
#         ax1.barh(
#             y=stage,
#             width=e - s,
#             left=s,
#             height=0.6,
#             color=op_colors[op],
#             edgecolor='black'
#         )
#         # Add operation + microbatch text inside the bar
#         ax1.text(
#             x=s + (e - s) / 2,
#             y=stage,
#             s=f"{op}{mb}",
#             va='center',
#             ha='center',
#             fontsize=10,
#             color='white',
#             fontweight='bold'
#         )

# ax1.set_ylabel("GPU")
# ax1.set_xlabel("Time")   # show x values
# ax1.set_yticks(range(1, p + 1))
# ax1.set_ylim(0.5, p + 0.5)
# ax1.set_title("GPU Operation Schedule\n(F_S=Forward Student, F_T=Forward Teacher, B=Backward, W=Weight Update)")
# ax1.grid(True, linestyle='--', alpha=0.4)
# handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
# ax1.legend(handles=handles, title='Operations', loc='upper right')

# # --- Bottom: Memory usage ---
# time_points = range(m * 4 + 10)

# for stage in range(1, p + 1):
#     events = [(s, delta_mem[op]) for s, _, _, op in sorted(schedule[stage], key=lambda x: x[0])]

#     mem_timeline = [0]
#     cur_mem = 0

#     for t in time_points:
#         while events and events[0][0] <= t:
#             _, delta = events.pop(0)
#             cur_mem += delta
#         mem_timeline.append(cur_mem)

#     ax2.plot(time_points, mem_timeline[:-1], label=f"GPU{stage}")

# ax2.set_xlabel("Time")
# ax2.set_ylabel("Memory (GB)")
# ax2.set_title("Per-GPU Memory Usage Over Time")
# ax2.grid(True, linestyle='--', alpha=0.4)
# ax2.legend()

# plt.tight_layout()
# plt.savefig("predef_zbts_2p_hand.png")
# plt.show()


# # --- Figure 2: Only Schedule (Gantt chart) ---
# fig2, ax_sched = plt.subplots(figsize=(14, 4))


# op_totals = {op: 0.0 for op in ops}
# for stage in range(1, p + 1):
#     for s, e, mb, op in sorted(schedule[stage]):
#         op_totals[op] += e - s

# for stage in range(1, p + 1):
#     for s, e, mb, op in sorted(schedule[stage]):
#         ax_sched.barh(
#             y=stage,
#             width=e - s,
#             left=s,
#             height=0.6,
#             color=op_colors[op],
#             edgecolor='black'
#         )
#         # Add operation + microbatch text
#         ax_sched.text(
#             x=s + (e - s) / 2,
#             y=stage,
#             s=f"{mb}",
#             va='center',
#             ha='center',
#             fontsize=10,
#             color='white',
#             fontweight='bold'
#         )

# ax_sched.set_ylabel("GPU")
# ax_sched.set_xlabel("Time")
# ax_sched.set_yticks(range(1, p + 1))
# ax_sched.set_ylim(0.5, p + 0.5)
# ax_sched.set_title("GPU Operation Schedule (Only)")
# ax_sched.grid(True, linestyle='--', alpha=0.4)

# handles2 = [patches.Patch(color=op_colors[c], label=f"{c}") for c in op_colors]
# legend_text = (
#     f"Makespan: {makespan:.2f}\n"
#     f"T: {T_total:.2f}\nα: {alpha:.2f} "
# )
# ax_sched.legend(handles=handles2, title=legend_text, loc='upper right')

# plt.tight_layout()
# plt.savefig("predef_zbts_2p_hand_schedule_only.png")
# plt.show()

