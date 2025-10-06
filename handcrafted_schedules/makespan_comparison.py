from zbts_p_1 import zbts_p_1, plot_schedule_and_memory
from zbts_2p import zbts_2p
from tspipe import tspipe
import numpy as np
import matplotlib.pyplot as plt



p = 16
MB_BBatch = 50
M_WBatch = 20
T_total = 10
alpha_values = np.arange(0.1, 5, 0.1)

tspipe_makespans = []
zbts_p_1_makespans = []
zbts_2p_makespans = []

for alpha in alpha_values:
    S, T, schedule, makespan, delta_mem = tspipe(p=p, m=p-1, M_BBatch=MB_BBatch, M_WBatch=M_WBatch, T_total=T_total, alpha=alpha)
    tspipe_makespans.append(makespan)
    
    S, T, schedule, makespan, delta_mem = zbts_p_1(p=p, m=p-1, M_BBatch=MB_BBatch, M_WBatch=M_WBatch, T_total=T_total, alpha=alpha)
    zbts_p_1_makespans.append(makespan)
    
    S, T, schedule, makespan, delta_mem = zbts_2p(p=p, m=2*p, M_BBatch=MB_BBatch, M_WBatch=M_WBatch, T_total=T_total, alpha=alpha)
    zbts_2p_makespans.append(makespan)
    


plt.plot(alpha_values, tspipe_makespans, label='TSPipe', marker='o')
plt.plot(alpha_values, zbts_p_1_makespans, label='ZBTS p-1', marker='o')
plt.plot(alpha_values, zbts_2p_makespans, label='ZBTS 2p', marker='o')
plt.xlabel('Alpha (Teacher Forward Time Multiplier)')
plt.ylabel('Makespan')
plt.title(f'Makespan vs Alpha (p={p}, m=p-1 for TSPipe and ZBTS p-1, m=2p for ZBTS 2p)')
plt.legend()
plt.grid(True)
plt.savefig("makespan_comparison.png")
plt.show()