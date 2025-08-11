import pulp


p = 4
m = 4
c_set = ['F', 'B', 'W']

T = {}
for i in range(1, p+1):
    for j in range(1, m+1):
        T[(i, j, 'F')] = 1
        T[(i, j, 'B')] = 1
        T[(i, j, 'W')] = 1

M_B, M_W = 25, 10
delta_M = {}
for i in range(1, p+1):
    for j in range(1, m+1):
        delta_M[(i, j, 'F')] = M_B
        delta_M[(i, j, 'B')] = M_W - M_B
        delta_M[(i, j, 'W')] = -M_W

M_limit = 12400
T_comm = 0

# Define Problem
prob = pulp.LpProblem("Pipeline_Scheduling", pulp.LpMinimize)

# Variables
E = pulp.LpVariable.dicts("E", [(i, j, c) for i in range(1, p+1)
                                for j in range(1, m+1)
                                for c in c_set], lowBound=T[(1, 1, 'F')])
O = pulp.LpVariable.dicts("O", [(i, j, c, i, jp, cp)
                                for i in range(1, p+1)
                                for j in range(1, m+1)
                                for c in c_set
                                for jp in range(1, m+1)
                                for cp in c_set], cat="Integer", lowBound=0, upBound=1)



# # fill O
for i in range(1, p+1):
    for j in range(1, m+1):
        for c in c_set:
            for jp in range(1, m+1):
                for cp in c_set:
                    
                    # If the microbatch is the same, do forward before backward and backward before weight update
                    if j == jp:
                        if c == "F" and cp == "B":
                            O[(i, j, c, i, jp, cp)].setInitialValue(1)
                            O[(i, j, c, i, jp, cp)].fixValue()
                            
                            O[(i, jp, cp, i, j, c)].setInitialValue(0)
                            O[(i, jp, cp, i, j, c)].fixValue()
                            
                        if c == "B" and cp == "W":
                            O[(i, j, c, i, jp, cp)].setInitialValue(1)
                            O[(i, j, c, i, jp, cp)].fixValue()
                            
                            O[(i, jp, cp, i, j, c)].setInitialValue(0)
                            O[(i, jp, cp, i, j, c)].fixValue()
                        
                        if c == "F" and cp == "W":
                            O[(i, j, c, i, jp, cp)].setInitialValue(1)
                            O[(i, j, c, i, jp, cp)].fixValue()
                            
                            O[(i, jp, cp, i, j, c)].setInitialValue(0)
                            O[(i, jp, cp, i, j, c)].fixValue()


                    # Given the same operation, do the previous microbatch first
                    if c == cp and j<=jp:
                        O[(i, j, c, i, jp, cp)].setInitialValue(1)
                        O[(i, j, c, i, jp, cp)].fixValue()
                    elif c == cp and j>jp:
                        O[(i, j, c, i, jp, cp)].setInitialValue(0)
                        O[(i, j, c, i, jp, cp)].fixValue()
                    
                    # O[(4, 1, "B", 4, 2, "F")].setInitialValue(1)
                    # O[(4, 1, "B", 4, 2, "F")].fixValue()
                    
                    # O[(4, 2, "F", 4, 1, "B")].setInitialValue(0)
                    # O[(4, 2, "F", 4, 1, "B")].fixValue()
                    
                            
                    


# Objective https://stackoverflow.com/questions/46319467/can-i-make-a-min-z-maxa-b-c-in-pulp
Z = pulp.LpVariable("Max_Stage_Completion", lowBound=0)
prob += Z

for i in range(1, p + 1):
    prob += Z >= E[(i, m, 'W')] - E[(i, 1, 'F')] + T[(i, 1, 'F')]

# @note : for two operation where the microbatches are different, only one is done before the other
for i in range(1, p + 1):
    for j in range(1, m + 1):
        for c in c_set:
            for jp in range(1, m + 1):
                for cp in c_set:
                    # !critical : find a way to make the matrix symetric without the problem being infeasable.
                    if j != jp or c != cp:
                        print(f"Adding constraint for O[{i}, {j}, {c}, {i}, {jp}, {cp}]")
                        prob += O[(i, j, c, i, jp, cp)] + O[(i, jp, cp, i, j, c)] == 1


# Constraints
for i in range(1, p+1):
    for j in range(1, m+1):
        if i > 1:
            prob += E[(i, j, 'F')] >= E[(i-1, j, 'F')] + T_comm + T[(i, j, 'F')] # forward rank dependency
        elif j==1:
            prob += E[(i, j, 'F')] == T[(i, j, 'F')]
        
        if i < p:
            prob += E[(i, j, 'B')] >= E[(i+1, j, 'B')] + T_comm + T[(i, j, 'B')] # backward rank dependency

for (i,j,c) in E:
    for (ip,jp,cp) in E:
        if i == ip:
            prob += E[(i, j, c)] >= E[(i, jp, cp)] + T[(i, j, c)] - O[(i,j,c,i,jp,cp)] * 1e20 # prevent overlap

                        
# for (i,jp,cp) in E:
#     prob += M_limit >= delta_M[(i, jp, cp)] + pulp.lpSum(delta_M[(i, j, c)] * O[(i, j, c, i, jp, cp)]
#                                                   for j in range(1, m+1)
#                                                   for c in c_set)

# Solve
prob.solve()
print("Status:", pulp.LpStatus[prob.status])

schedule = {}
before = {}
max_time = 0
for v in prob.variables():
    # print(f"{v.name} = {v.varValue}")
    
    if "O" in v.name:
        before[v.name] = v.varValue
    
    if "Max" in v.name:
        max_time = v.varValue
        print(f"Max Stage Completion Time: {max_time}")
    
    if "E" in v.name:
        schedule[v.name] = v.varValue
        # print(f"{v.name} = {v.varValue}")

# node 4, (1,b) vs (2,f)
# print(before["O_(4,_1,_'B',_4,_2,_'F')"])
# print(before["O_(4,_2,_'F',_4,_1,_'B')"])
# print(f"{schedule["E_(1,_1,_"F")"]} >= {schedule["E_(1,_2,_"F")"]} + 1 - {before["O_(1,_1,_'F',_1,_2,_'F')"]}")
# print(f"{schedule['E_(1,_2,_'F')']} >= {schedule['E_(1,_1,_'F')']} + 1 - {before['O_(1,_2,_'F',_1,_1,_'F')']}")


# E_(1,_2,_'F'): 1.0
# E_(1,_2,_'W'): 1.0
# print(before["O_(1,_2,_'F',_1,_2,_'W')"])
# print(before["O_(1,_2,_'W',_1,_2,_'F')"])


print(" comp 1 vs 2")
print(before["O_(1,_1,_'F',_1,_2,_'F')"])
print(before["O_(1,_2,_'F',_1,_1,_'F')"])

print(" comp 2 vs 3")
print(before["O_(1,_2,_'F',_1,_3,_'F')"])
print(before["O_(1,_3,_'F',_1,_2,_'F')"])

print(" comp 3 vs 4")
print(before["O_(1,_3,_'F',_1,_4,_'F')"])
print(before["O_(1,_4,_'F',_1,_3,_'F')"])

print(" comp 4 vs 1")
print(before["O_(1,_4,_'F',_1,_1,_'F')"])
print(before["O_(1,_1,_'F',_1,_4,_'F')"])

print(" comp 1 vs 3")
print(before["O_(1,_1,_'F',_1,_3,_'F')"])
print(before["O_(1,_3,_'F',_1,_1,_'F')"])

# 4 and 2
print(" comp 2 vs 4")
print(before["O_(1,_2,_'F',_1,_4,_'F')"])
print(before["O_(1,_4,_'F',_1,_2,_'F')"])

sorted_keys = sorted(schedule.keys(), key=lambda k: schedule[k])

# Print the keys in order of their values
for key in sorted_keys:
    print(f"{key}: {schedule[key]}")
    
# Print elements of delta_M
# for key, value in delta_M.items():
#     print(f"delta_M[{key}] = {value}")




import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

op_colors = {
    'F': 'blue',
    'B': 'red',
    'W': 'green'
}

fig, ax = plt.subplots(figsize=(12, 6))

rectangles = []
for key, end_time in schedule.items():
    # Parse GPU (i), microbatch (j), and operation (c)
    parts = key.split('_')
    i = int(parts[1].strip('(,'))
    j = int(parts[2].strip(',)'))
    c = parts[3].strip("'()")
    
    # print(f"GPU: {i}, Microbatch: {j}, Operation: {c}, End Time: {end_time}")
    
    start_time = end_time - 1 # @param
    
    # Create rectangle (x, y, width, height)
    rect = patches.Rectangle(
        (start_time, i - 0.4),  # (x, y)
        1.0,  # width (duration)
        0.8,  # height (GPU height)
        facecolor=op_colors[c],
        edgecolor='black',
        label=f'{c}'
    )
    # print(f"Rectangle: {rect}")
    rectangles.append(rect)
    
    
    ax.text(start_time + 0.5, i, f'{c}{j}', ha='center', va='center', color='white')


pc = PatchCollection(rectangles, match_original=True)
ax.add_collection(pc)

ax.set_xlabel('Time')
ax.set_ylabel('GPU')
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(['GPU 1', 'GPU 2', 'GPU 3', 'GPU 4'])
ax.set_title('GPU Operation Schedule (F=Forward, B=Backward, W=Weight Update)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(0, 25 + 1) # @param max_time
ax.set_ylim(0, p+1)

handles = [patches.Patch(color=op_colors[c], label=c) for c in op_colors]
ax.legend(handles=handles, title='Operations')

plt.tight_layout()
plt.show()