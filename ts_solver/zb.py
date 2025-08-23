from collections import defaultdict
import pulp
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def precedes(a, b, y):
    # Check if task a precedes task b
    
    if a == b:
        return 1
    
    if (a, b) in y:
        return y[(a, b)]
    elif (b, a) in y:
        return 1 - y[(b, a)]
    else:
        raise ValueError(f"No precedence relation defined for {a} and {b}")
        



def schedule_ts(p=3, m=3, T_comm=0.0, gpu_mem_limit=100, delta_mem=None, time_limit=60*10, msg=0):
    if delta_mem is None:
        delta_mem = {'F': M_B, 'B': M_W - M_B, 'W': -M_W}

    T = {}
    for stage in range(1, p+1):
        for mb in range(1, m+1):
            T[(stage, mb, 'F')] = 1
            T[(stage, mb, 'B')] = 1
            T[(stage, mb, 'W')] = 1

    mdl = pulp.LpProblem("ZB_ILP_fixed_ordered", pulp.LpMinimize)

    S = {k: pulp.LpVariable(f"S_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}
    E = {k: pulp.LpVariable(f"E_{k[0]}_{k[1]}_{k[2]}", lowBound=0) for k in T}

    # Objective
    Z = pulp.LpVariable("Z", lowBound=0)
    for stage in range(1, p+1):
        mdl += Z >= E[(stage, m, 'W')] - S[(stage, 1, 'F')]
    mdl += Z

    # start end relation
    for k, dur in T.items():
        mdl += E[k] == S[k] + dur

    # F -> B -> W
    for stage in range(1, p+1):
        for mb in range(1, m+1):
            mdl += S[(stage, mb, 'B')] >= E[(stage, mb, 'F')] + T_comm
            mdl += S[(stage, mb, 'W')] >= E[(stage, mb, 'B')] + T_comm

    # microbatch order
    ops = ['F','B','W']
    for stage in range(1, p+1):
        for op in ops:
            for j in range(2, m+1):
                mdl += S[(stage, j, op)] >= E[(stage, j-1, op)] + T_comm

    # Forward dependency
    for stage in range(2, p+1):
        for mb in range(1, m+1):
            mdl += S[(stage, mb, 'F')] >= E[(stage-1, mb, 'F')] + T_comm

    # Backward dependency
    for stage in range(1, p):
        for mb in range(1, m+1):
            mdl += S[(stage, mb, 'B')] >= E[(stage+1, mb, 'B')] + T_comm
        

    M = 1e5
    y = {}
    for stage in range(1, p+1):
        tasks_on_stage = [task for task in T.keys() if task[0] == stage]
        for a in tasks_on_stage:
            for b in tasks_on_stage:
                if (a,b) not in y and (b,a) not in y:
                    # y(a,b) means a comes before b therefor E(a) <= E(b)
                    y[(a, b)] = pulp.LpVariable(f"y_{a}_{b}", lowBound=0, upBound=1, cat="Binary")
                    mdl += E[a] >= E[b] + T[a] - M * precedes(a, b, y)
                    mdl += E[b] >= E[a] + T[b] - M * precedes(b, a, y)
                    # mdl += S[a] >= E[b] - M * precedes(a, b, y)
                    # mdl += S[b] >= E[a] - M * precedes(b, a, y)




    # Memory limit constraint

    for stage in range(1, p+1):
        tasks_stage = [t for t in T.keys() if t[0] == stage]

        # Enforce capacity at every finish instant E[t]
        for b in tasks_stage:
            # Sum deltas of all tasks u whose E[a] <= E[b]
            mem_prefix_terms = []
            for a in tasks_stage:
                # print(f"Checking precedes relation for {a} and {b}")
                mem_prefix_terms.append(delta_mem[a[2]] * precedes(a, b, y))

            mdl += pulp.lpSum(mem_prefix_terms) <= gpu_mem_limit

    # Solve the problem
    mdl.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    print("Status:", pulp.LpStatus[mdl.status])
    
    print("Objective value (Z):", pulp.value(Z))
    
    
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
        bubble_sizes[stage] = total_time - task_time
        total_bubble_size += bubble_sizes[stage]
    
    return total_bubble_size, bubble_sizes, schedule, total_time




def plot_schedule(mdl, Z, S, E, T, y, p, delta_mem, schedule):
    # # Remove previous image if it exists
    # if os.path.exists("zb.png"):
    #     os.remove("zb.png")
    
  

    # Plot schedule
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # --- Top: Gantt chart ---
    op_colors = {'F': 'royalblue', 'B': 'crimson', 'W': 'forestgreen'}
    for stage in range(1, p+1):
        for s, e, mb, op in sorted(schedule[stage]):
            ax1.barh(stage, e - s, left=s, height=0.6,
                     color=op_colors[op], edgecolor='black')
            ax1.text(s + (e - s) / 2, stage, f"{op}{mb}",
                     va='center', ha='center', fontsize=8, color='white')

    ax1.set_ylabel("GPU")
    ax1.set_yticks(range(1, p+1))
    ax1.set_ylim(0.5, p + 0.5)
    ax1.set_title("GPU Operation Schedule (F=Forward, B=Backward, W=Weight Update)")
    ax1.grid(True, linestyle='--', alpha=0.4)
    handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
    ax1.legend(handles=handles, title='Operations', loc='upper right')

    # --- Bottom: Memory usage ---
    time_points = range(int(pulp.value(Z)) + 5)
    for stage in range(1, p+1):
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
    plt.savefig("zb.png")




def analyze_bubble_vs_gpu_limit(p, m, T_comm, M_B, M_W, delta_mem, time_limit=60*10):
    gpu_limits = range(M_B, int(2 * p * M_B + 50), 5)
    bubble_ratios = []
    total_times = []

    for gpu_mem_limit in gpu_limits:
        mdl, Z, S, E, T, y = schedule_ts(p=p, m=m, T_comm=T_comm, gpu_mem_limit=gpu_mem_limit, delta_mem=delta_mem, time_limit=time_limit)
        
        if pulp.LpStatus[mdl.status] != "Optimal":
            bubble_ratios.append(None)
            total_times.append(None)
            continue

        tot_bubble_size, bubble_sizes, schedule, total_time = bubble_info(mdl, Z, S, E, T, y, p)
        
        print(f"GPU Memory Limit: {gpu_mem_limit} GB")
        print(f"Total time (Z): {total_time}")
        print(f"Total bubble size: {tot_bubble_size}")
        print(f"Bubble ratio: {tot_bubble_size / (total_time * p):.2f}")
        plot_schedule(mdl, Z, S, E, T, y, p, delta_mem, schedule)
        
        # import pdb; pdb.set_trace()
        
        bubble_ratios.append(tot_bubble_size / (total_time*p))
        total_times.append(total_time)

    # Plot bubble ratio vs GPU memory limit
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(gpu_limits, bubble_ratios, marker='o', label="Bubble Ratio")
    plt.xlabel("GPU Memory Limit (GB)")
    plt.ylabel("Bubble Ratio")
    plt.title("Bubble Ratio vs GPU Memory Limit")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    # Plot total time vs GPU memory limit
    plt.subplot(1, 2, 2)
    plt.plot(gpu_limits, total_times, marker='o', label="Total Time", color="orange")
    plt.xlabel("GPU Memory Limit (GB)")
    plt.ylabel("Total Time")
    plt.title("Total Time vs GPU Memory Limit")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig("bubble_vs_gpu_limit.png")
    # plt.show()



if __name__ == "__main__":
    
    p = 3                   # @param GPUs
    m = 3                   # @param microbatches
    T_comm = 0.0            # @param inter-stage communication time
    M_B, M_W = 25, 10       # @param Memory usage for B and W operations in GB
    gpu_mem_limit = p*M_B+5    # @param GPU memory limit in GB
    delta_mem = {'F': M_B, 'B': M_W - M_B, 'W': -M_W}
    
    mdl, Z, S, E, T, y = schedule_ts(p=p, m=m, T_comm=T_comm, gpu_mem_limit=gpu_mem_limit, delta_mem=delta_mem)

    # Check if the problem is infeasible
    if pulp.LpStatus[mdl.status] != "Optimal":
        print("The problem is infeasible or could not be solved optimally.")
        exit(1)


    tot_bubble_size, bubble_sizes, schedule, total_time = bubble_info(mdl, Z, S, E, T, y, p)

    print(f"Total time (Z): {total_time}")
    print(f"Total bubble size: {tot_bubble_size}")
    print(f"Bubble ratio: {tot_bubble_size / (total_time*p):.2f}")
    
    plot_schedule(mdl, Z, S, E, T, y, p, delta_mem, schedule)
    
    
    # analyze_bubble_vs_gpu_limit(p, m, T_comm, M_B, M_W, delta_mem, time_limit=60*10)
