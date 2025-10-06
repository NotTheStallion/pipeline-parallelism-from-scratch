from collections import defaultdict
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches




def precedes(a, b, y):
    if a == b:
        return 1
    
    if (a, b) in y:
        return y[(a, b)]
    elif (b, a) in y:
        return 1 - y[(b, a)]
    else:
        raise ValueError(f"No precedence relation defined for {a} and {b}")
    

import pulp
from collections import defaultdict


def schedule_ts(p=3, m=3, T_comm=0.0, gpu_mem_limit=100, delta_mem=None, time_limit=60*10, msg=0):
    alpha = 1.5
    
    T = {}
    for stage in range(1, p+1):
        for mb in range(1, m+1):
            T[(stage, mb, 'F_S')] = 1
            T[(stage, mb, 'F_T')] = alpha * T[(1, 1, 'F_S')]
            T[(stage, mb, 'B')] = 1
            T[(stage, mb, 'W')] = 1

    mdl = pulp.LpProblem("ZB_ILP_fixed_ordered", pulp.LpMinimize)

    S = {k: pulp.LpVariable(f"S_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
    E = {k: pulp.LpVariable(f"E_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}

    # Objective
    Z = pulp.LpVariable("Z", lowBound=0)
    # for stage in range(1, p+1):
    #     mdl += Z >= E[(stage, m, 'W')] - min(S[(stage, 1, 'F_S')], S[(stage, 1, 'F_T')])
    # mdl += Z


    for stage in range(1, p + 1):
        W = pulp.LpVariable(f"W_{stage}", lowBound=0)

        S1 = S[(stage, 1, 'F_S')]
        S2 = S[(stage, 1, 'F_T')]

        mdl += W <= S1
        mdl += W <= S2

        mdl += Z >= E[(stage, m, 'W')] - W
    mdl += Z

    # start end relation
    for k, dur in T.items():
        mdl += E[k] == S[k] + dur

    # F -> B -> W
    for stage in range(1, p+1):
        for mb in range(1, m+1):
            mdl += S[(stage, mb, 'B')] >= E[(stage, mb, 'F_S')]
            mdl += S[(stage, mb, 'B')] >= E[(stage, mb, 'F_T')]
            mdl += S[(stage, mb, 'W')] >= E[(stage, mb, 'B')]

    # microbatch order
    ops = ['F_S','F_T','B','W']
    for stage in range(1, p+1):
        for op in ops:
            for j in range(2, m+1):
                mdl += S[(stage, j, op)] >= E[(stage, j-1, op)]

    # Forward dependency
    for stage in range(2, p+1):
        for mb in range(1, m+1):
            mdl += S[(stage, mb, 'F_S')] >= E[(stage-1, mb, 'F_S')] + T_comm
            mdl += S[(stage, mb, 'F_T')] >= E[(stage-1, mb, 'F_T')] + T_comm

    # Backward dependency
    for stage in range(1, p):
        for mb in range(1, m+1):
            mdl += S[(stage, mb, 'B')] >= E[(stage+1, mb, 'B')] + T_comm


    M = 1e5 # horizon_upper_()bound
    y = {}
    for stage in range(1, p+1):
        tasks_on_stage = [task for task in T.keys() if task[0] == stage]
        for a in tasks_on_stage:
            for b in tasks_on_stage:
                if (a,b) not in y and (b,a) not in y:
                    y[(a, b)] = pulp.LpVariable(f"y_{a}_{b}", lowBound=0, upBound=1, cat="Binary")
                    mdl += E[a] >= E[b] + T[a] - M * precedes(a, b, y)
                    mdl += E[b] >= E[a] + T[b] - M * precedes(b, a, y)




    # Memory limit constraint


    for stage in range(1, p+1):
        tasks_stage = [t for t in T.keys() if t[0] == stage]

        # Enforce capacity at every finish instant E[t]
        for b in tasks_stage:
            # Sum deltas of all tasks u whose E[a] <= E[b]
            mem_prefix_terms = []
            for a in tasks_stage:
                mem_prefix_terms.append(delta_mem[a[2]] * precedes(a, b, y))

            mdl += pulp.lpSum(mem_prefix_terms) <= gpu_mem_limit





    mdl.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    print("Status:", pulp.LpStatus[mdl.status])
    print("Objective (Z):", pulp.value(Z))
    
    return mdl, Z, S, E, T, y



def bubble_info(mdl, Z, S, E, T, y, p):
    schedule = defaultdict(list)
    
      
    for task in sorted(T.keys()):
        # task = (stage, mb, op)
        s = float(pulp.value(S[task]))
        e = float(pulp.value(E[task]))
        schedule[task[0]].append((s, e, task[1], task[2]))

    print(schedule[2])
    
    # Compute bubble size (idle time) per stage
    total_time = float(pulp.value(Z))
    bubble_sizes = {}
    total_bubble_size = 0
    for stage in range(1, p+1):
        # Calculate the total time for the stage
        
        # Count the time spent on tasks (F, B, W) for this stage
        task_time = sum(
            float(pulp.value(E[task]) - pulp.value(S[task]))
            for task in T.keys() if task[0] == stage
        )
        
        # Subtract the time spent on tasks from the total time to get idle time
        bubble_sizes[stage] = total_time - task_time - 2 # @param substract two for future teacher forwards
        total_bubble_size += bubble_sizes[stage]
    
    return total_bubble_size, bubble_sizes, schedule, total_time








def plot_schedule(mdl, Z, S, E, T, y, p, m, delta_mem, schedule):
    # Plot schedule (Teacher-Student)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # --- Top: Gantt chart ---
    op_colors = {'F_S': 'royalblue', 'F_T': 'orange', 'B': 'crimson', 'W': 'forestgreen'}
    for stage in range(1, p + 1):
        for s, e, mb, op in sorted(schedule[stage]):
            ax1.barh(stage, (e - s) / (m / 2), left=s / (m / 2), height=0.6,
                     color=op_colors[op], edgecolor='black')
            ax1.text((s + (e - s) / 2) / (m / 2), stage, f"{op}{mb}",
                     va='center', ha='center', fontsize=12, color='white')  # Increased fontsize

    ax1.set_ylabel("GPU", fontsize=12)  # Increased fontsize
    ax1.set_yticks(range(1, p + 1))
    ax1.set_ylim(0.5, p + 0.5)
    ax1.set_title("GPU Operation Schedule (F=Forward, B=Backward, W=Weight Update)", fontsize=14)  # Increased fontsize
    ax1.grid(True, linestyle='--', alpha=0.4)
    handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
    ax1.legend(handles=handles, title='Operations', loc='upper right', fontsize=10, title_fontsize=12)  # Increased fontsize

    # --- Bottom: Memory usage ---
    time_points = range(int(pulp.value(Z)) + 5)
    time_points = range(15)
    for stage in range(1, p + 1):
        events = []
        for s, e, mb, op in sorted(schedule[stage], key=lambda x: x[0]):
            events.append((s / (m / 2), delta_mem[op]))
        mem_timeline = [0]
        cur_mem = 0
        last_t = 0
        for t in time_points:
            t /= (m / 2)
            while events and events[0][0] <= t:
                _, delta = events.pop(0)
                cur_mem += delta
            mem_timeline.append(cur_mem)
        scaled_mem_timeline = [mem / (m / 2) for mem in mem_timeline[:-1]]
        ax2.plot([tp / (m / 2) for tp in time_points], scaled_mem_timeline, label=f"GPU{stage}")

    ax2.set_xlabel(f"Time [{1/(m/2):.2f} per op]", fontsize=12)  # Increased fontsize
    ax2.set_ylabel("Memory (GB, scaled)", fontsize=12)  # Increased fontsize
    ax2.set_title("Per-GPU Memory Usage Over Time (Scaled)", fontsize=14)  # Increased fontsize
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=13)  # Increased fontsize

    plt.tight_layout()
    plt.savefig("ts_zb.png")



def analyze_bubble_vs_gpu_limit(p, m, T_comm, M_B, M_W, delta_mem, time_limit=60*10):
    gpu_limits = range(M_B, int(2 * p * M_B + 50), 5)
    bubble_ratios = []
    total_times = []
    
    tspipe_bubble_ratio = 2/12
    tspipe_makespan = 12-2
    tspipe_br = []
    tspipe_ms = []

    for gpu_mem_limit in gpu_limits:
        mdl, Z, S, E, T, y = schedule_ts(p=p, m=m, T_comm=T_comm, gpu_mem_limit=gpu_mem_limit, delta_mem=delta_mem, time_limit=time_limit)
        
        if gpu_mem_limit > m*M_B:
            tspipe_br.append(tspipe_bubble_ratio)
            tspipe_ms.append(tspipe_makespan)
        else:
            tspipe_br.append(0)
            tspipe_ms.append(0)
        
        if pulp.LpStatus[mdl.status] != "Optimal":
            bubble_ratios.append(None)
            total_times.append(None)
            continue

        tot_bubble_size, bubble_sizes, schedule, total_time = bubble_info(mdl, Z, S, E, T, y, p)
        
        print(f"GPU Memory Limit: {gpu_mem_limit} GB")
        print(f"Total time (Z): {total_time}")
        print(f"Total bubble size: {tot_bubble_size}")
        print(f"Bubble ratio: {tot_bubble_size / (total_time * p):.2f}")
        plot_schedule(mdl, Z, S, E, T, y, p, m, delta_mem, schedule)
        
        bubble_ratios.append(tot_bubble_size / (total_time * p))
        total_times.append(total_time)

    # Plot bubble ratio vs GPU memory limit
    plt.figure(figsize=(12, 6))
    
    # Filter out points where tspipe_br or tspipe_ms are 0
    valid_indices = [i for i in range(len(tspipe_br)) if tspipe_br[i] != 0]
    valid_gpu_limits = [gpu_limits[i] for i in valid_indices]
    valid_tspipe_br = [tspipe_br[i] for i in valid_indices]
    valid_tspipe_ms = [tspipe_ms[i] for i in valid_indices]

    # Plot bubble ratio vs GPU memory limit
    plt.subplot(1, 2, 1)
    plt.plot(gpu_limits, bubble_ratios, marker='o', label="Zero Bubble Bubble Ratio", color="green")
    plt.plot(valid_gpu_limits, valid_tspipe_br, linestyle='--', label="TSPipe Bubble Ratio", color="blue")
    plt.xlabel("GPU Memory Limit (GB)")
    plt.ylabel("Bubble Ratio")
    plt.title("Bubble Ratio vs GPU Memory Limit")
    plt.ylim(0, 1)  # Set the y-axis limits for bubble ratio
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    # Plot Makespan vs GPU memory limit
    plt.subplot(1, 2, 2)
    plt.plot(gpu_limits, total_times, marker='o', label="ZBTS Makespan", color="green")
    plt.plot(valid_gpu_limits, valid_tspipe_ms, linestyle='--', label="TSPipe Makespan", color="blue")
    plt.xlabel("GPU Memory Limit (GB)")
    plt.ylabel("Makespan")
    plt.title("Makespan vs GPU Memory Limit")
    plt.ylim(m*4, 15)  # Set the y-axis limits for makespan
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig("bubble_vs_gpu_limit_ts.png")


if __name__ == "__main__":
    p = 2                   # @param GPUs
    m = 2*p                  # @param microbatches
    T_comm = 0.0            # @param inter-stage communication time
    M_B, M_W = 25, 10       # @param Memory usage for B and W operations in GB
    gpu_mem_limit = m*M_B   # @param GPU memory limit in GB
    delta_mem = {'F_S': M_B, 'F_T':0, 'B': M_W - M_B, 'W': -M_W}
    
    # special
    mdl, Z, S, E, T, y = schedule_ts(p=p, m=m, T_comm=T_comm, gpu_mem_limit=gpu_mem_limit, delta_mem=delta_mem, time_limit=60*10, msg=1)
    
    
    tot_bubble_size, bubble_sizes, schedule, total_time = bubble_info(mdl, Z, S, E, T, y, p)
    
    # schedule = defaultdict(list)
    # for task in sorted(T.keys()):
    #     # task = (stage, mb, op)
    #     s = float(pulp.value(S[task]))
    #     e = float(pulp.value(E[task]))
    #     schedule[task[0]].append((s, e, task[1], task[2]))

    # print(schedule[1])
    
    plot_schedule(mdl, Z, S, E, T, y, p, m, delta_mem, schedule)
    
    # analyze_bubble_vs_gpu_limit(p, m, T_comm, M_B, M_W, delta_mem, time_limit=60*10)
    
    