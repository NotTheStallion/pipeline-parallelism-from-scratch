import numpy as np
import matplotlib.pyplot as plt

# Number of pipeline stages (p)
p = np.arange(1, 5, 0.1)  # p from 1 to 4.9

# TFT is a constant, can set to 1 for visualization
TFT = 1

# Makespan formulas
makespanTSPipe = 5 * (p - 1) * TFT
makespanZBTS = 3 * (p - 1) * TFT + p * TFT

plt.figure(figsize=(8, 5))
plt.plot(p, makespanTSPipe, label='makespanTSPipe = 5(p-1)TFT')
plt.plot(p, makespanZBTS, label='makespanZBTS = 3(p-1)TFT + pTFT')

mask = makespanTSPipe > makespanZBTS
plt.fill_between(p, makespanTSPipe, makespanZBTS, where=mask, color='red', alpha=0.3, label='TSPipe > ZBTS')

p_proj = 3
y_tspipe = 5 * (p_proj - 1) * TFT
y_zbts = 3 * (p_proj - 1) * TFT + p_proj * TFT

plt.axvline(x=p_proj, color='gray', linestyle='--', alpha=0.7)
plt.scatter([p_proj], [y_tspipe], color='blue')
plt.scatter([p_proj], [y_zbts], color='orange')
plt.text(p_proj + 0.1, y_tspipe, f'makespan {y_tspipe}', color='blue', va='bottom')
plt.text(p_proj + 0.1, y_zbts, f'makespan {y_zbts}', color='orange', va='top')

plt.xlabel('Number of pipeline stages (p)')
plt.ylabel('Makespan')
plt.title('Comparison of Makespans')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Set integer ticks for x and y axes
plt.xticks(np.arange(1, 6, 1))
plt.yticks(np.arange(0, int(max(makespanTSPipe.max(), makespanZBTS.max())) + 2, 1))

plt.savefig('makespan_comparison.png')