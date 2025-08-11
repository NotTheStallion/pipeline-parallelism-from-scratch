import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def solve_pipeline_schedule(p, m, T, delta_M, M_limit, T_comm=0):
    """
    Solve pipeline scheduling problem using ILP.
    
    Args:
        p: Number of pipeline stages (GPUs)
        m: Number of microbatches
        T: Time costs dict {(stage, microbatch, pass_type): time}
        delta_M: Memory increments dict {(stage, microbatch, pass_type): memory_change}
        M_limit: Maximum memory limit
        T_comm: Communication time between stages
    """
    
    c_set = ['F', 'B', 'W']  # Pass types: Forward, Backward, Weight update
    
    # Create problem
    prob = pulp.LpProblem("Pipeline_Scheduling", pulp.LpMinimize)
    
    # Variables: Ending times E[i,j,c]
    E = pulp.LpVariable.dicts("E", 
                             [(i, j, c) for i in range(1, p+1)
                              for j in range(1, m+1) 
                              for c in c_set], 
                             lowBound=T[(1, 1, 'F')])
    
    # Variables: Ordering O[i,j,c,i,jp,cp] = 1 if (i,j,c) before (i,jp,cp)
    O = pulp.LpVariable.dicts("O", 
                             [(i, j, c, i, jp, cp)
                              for i in range(1, p+1)
                              for j in range(1, m+1)
                              for c in c_set
                              for jp in range(1, m+1)
                              for cp in c_set], 
                             cat="Binary")
    
    # Fix ordering for same microbatch: F -> B -> W
    for i in range(1, p+1):
        for j in range(1, m+1):
            # F before B
            O[(i, j, 'F', i, j, 'B')].setInitialValue(1)
            O[(i, j, 'F', i, j, 'B')].fixValue()
            O[(i, j, 'B', i, j, 'F')].setInitialValue(0)
            O[(i, j, 'B', i, j, 'F')].fixValue()
            
            # B before W
            O[(i, j, 'B', i, j, 'W')].setInitialValue(1)
            O[(i, j, 'B', i, j, 'W')].fixValue()
            O[(i, j, 'W', i, j, 'B')].setInitialValue(0)
            O[(i, j, 'W', i, j, 'B')].fixValue()
            
            # F before W
            O[(i, j, 'F', i, j, 'W')].setInitialValue(1)
            O[(i, j, 'F', i, j, 'W')].fixValue()
            O[(i, j, 'W', i, j, 'F')].setInitialValue(0)
            O[(i, j, 'W', i, j, 'F')].fixValue()
    
    # Objective: Minimize maximum stage completion time
    Z = pulp.LpVariable("Max_Stage_Completion", lowBound=0)
    prob += Z
    
    for i in range(1, p + 1):
        prob += Z >= E[(i, m, 'W')] - E[(i, 1, 'F')] + T[(i, 1, 'F')]
    
    # Ordering constraints: Each pair must have exactly one ordering
    for i in range(1, p + 1):
        for j in range(1, m + 1):
            for c in c_set:
                for jp in range(1, m + 1):
                    for cp in c_set:
                        if j != jp or c != cp:
                            prob += O[(i, j, c, i, jp, cp)] + O[(i, jp, cp, i, j, c)] == 1
    
    # Sequential constraints
    for i in range(1, p+1):
        for j in range(1, m+1):
            # Forward pass dependency
            if i > 1:
                prob += E[(i, j, 'F')] >= E[(i-1, j, 'F')] + T_comm + T[(i, j, 'F')]
            # elif j == 1:
            #     prob += E[(i, j, 'F')] == T[(i, j, 'F')]
            
            # Backward pass dependency
            if i < p:
                prob += E[(i, j, 'B')] >= E[(i+1, j, 'B')] + T_comm + T[(i, j, 'B')]
    
    # No overlap constraints
    big_M = 1e20
    for (i, j, c) in E:
        for (ip, jp, cp) in E:
            if i == ip and (j != jp or c != cp):
                prob += E[(i, j, c)] >= E[(i, jp, cp)] + T[(i, j, c)] - O[(i, j, c, i, jp, cp)] * big_M
    
    # Memory constraints
    # for (i, jp, cp) in E:
    #     memory_expr = delta_M[(i, jp, cp)]
    #     for j in range(1, m+1):
    #         for c in c_set:
    #             if (j, c) != (jp, cp):
    #                 memory_expr += delta_M[(i, j, c)] * O[(i, j, c, i, jp, cp)]
    #     prob += memory_expr <= M_limit
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract results
    if prob.status == pulp.LpStatusOptimal:
        schedule = {}
        ordering = {}
        max_time = 0
        
        for v in prob.variables():
            if "O" in v.name and v.varValue == 1:
                ordering[v.name] = v.varValue
            elif "Max" in v.name:
                max_time = v.varValue
            elif "E" in v.name:
                schedule[v.name] = v.varValue
        
        return {
            'status': 'Optimal',
            'max_time': max_time,
            'schedule': schedule,
            'ordering': ordering
        }
    else:
        return {'status': pulp.LpStatus[prob.status]}


def plot_schedule(result, p, T):
    """Plot the pipeline schedule as a Gantt chart"""
    if result['status'] != 'Optimal':
        print(f"Cannot plot: {result['status']}")
        return
    
    op_colors = {
        'F': 'blue',
        'B': 'red',
        'W': 'green'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rectangles = []
    
    for key, end_time in result['schedule'].items():
        # Parse stage (i), microbatch (j), and operation (c)
        parts = key.split('_')
        i = int(parts[1].strip('(,'))
        j = int(parts[2].strip(',)'))
        c = parts[3].strip("'()")
        
        # Calculate start time
        duration = T.get((i, j, c), 1)
        start_time = end_time - duration
        
        # Create rectangle
        rect = patches.Rectangle(
            (start_time, i - 0.4),
            duration,
            0.8,
            facecolor=op_colors[c],
            edgecolor='black',
            alpha=0.8
        )
        rectangles.append(rect)
        
        # Add text label
        ax.text(start_time + duration/2, i, f'{c}{j}', 
               ha='center', va='center', color='white', fontweight='bold')
    
    # Add rectangles to plot
    for rect in rectangles:
        ax.add_patch(rect)
    
    # Configure plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Pipeline Stage')
    ax.set_yticks(range(1, p+1))
    ax.set_yticklabels([f'Stage {i}' for i in range(1, p+1)])
    ax.set_title('Pipeline Schedule (F=Forward, B=Backward, W=Weight Update)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 40) # @param result['max_time'] + 2
    ax.set_ylim(0, p+1)
    
    # Add legend
    handles = [patches.Patch(color=op_colors[c], 
               label=f'{c} ({"Forward" if c=="F" else "Backward" if c=="B" else "Weight Update"})') 
               for c in op_colors]
    ax.legend(handles=handles, title='Operations')
    
    plt.tight_layout()
    plt.show()


def print_schedule(result):
    """Print the schedule in a readable format"""
    if result['status'] != 'Optimal':
        print(f"Status: {result['status']}")
        return
    
    print(f"Max Stage Completion Time: {result['max_time']:.2f}")
    print("\nSchedule (sorted by completion time):")
    print("Variable Name : End Time")
    print("-" * 40)
    
    # Sort by end time
    sorted_schedule = sorted(result['schedule'].items(), key=lambda x: x[1])
    for key, end_time in sorted_schedule:
        print(f"{key}: {end_time}")


def run_example():
    """Run the example from your original code"""
    # Problem parameters
    p = 4  # stages
    m = 4  # microbatches
    
    # Time costs
    T = {}
    for i in range(1, p+1):
        for j in range(1, m+1):
            T[(i, j, 'F')] = 1
            T[(i, j, 'B')] = 1
            T[(i, j, 'W')] = 1
    
    # Memory parameters
    M_B, M_W = 25, 10
    delta_M = {}
    for i in range(1, p+1):
        for j in range(1, m+1):
            delta_M[(i, j, 'F')] = M_B
            delta_M[(i, j, 'B')] = M_W - M_B
            delta_M[(i, j, 'W')] = -M_W
    
    M_limit = 12400
    T_comm = 0
    
    # Solve
    result = solve_pipeline_schedule(p, m, T, delta_M, M_limit, T_comm)
    
    # Print results
    print_schedule(result)
    
    # Plot results
    plot_schedule(result, p, T)
    
    return result


def run_simple_example():
    """Run a simpler example for testing"""
    # Smaller problem for debugging
    p = 2  # stages  
    m = 2  # microbatches
    
    # Time costs
    T = {}
    for i in range(1, p+1):
        for j in range(1, m+1):
            T[(i, j, 'F')] = 2
            T[(i, j, 'B')] = 2  
            T[(i, j, 'W')] = 1
    
    # Memory parameters
    M_B, M_W = 10, 5
    delta_M = {}
    for i in range(1, p+1):
        for j in range(1, m+1):
            delta_M[(i, j, 'F')] = M_B
            delta_M[(i, j, 'B')] = M_W - M_B
            delta_M[(i, j, 'W')] = -M_W
    
    M_limit = 50
    T_comm = 0
    
    # Solve
    result = solve_pipeline_schedule(p, m, T, delta_M, M_limit, T_comm)
    
    # Print and plot
    print_schedule(result)
    plot_schedule(result, p, T)
    
    return result


if __name__ == "__main__":
    print("Running original example (4 stages, 4 microbatches):")
    result1 = run_example()
    
    # print("\n" + "="*60 + "\n")
    
    # print("Running simple example (2 stages, 2 microbatches):")
    # result2 = run_simple_example()