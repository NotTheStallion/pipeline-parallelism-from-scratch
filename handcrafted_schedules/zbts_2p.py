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
