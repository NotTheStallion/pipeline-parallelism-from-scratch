import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 3  # number of stages
M_B_batch = 25  # batch-level backward memory (arbitrary units)
M_W_batch = 10  # batch-level weight gradient memory (arbitrary units)

# Function to compute peak memory for given stage i
def peak_memory(i, m, M_B_batch, M_W_batch, p):
    return ((p - i) * M_B_batch + (m - (p - i)) * M_W_batch) / m

# Range of microbatches (start from 6)
microbatches = np.arange(6, 21)  # m = 6 to 20

# Compute peak memory for each stage
peak_stage1 = [peak_memory(1, m, M_B_batch, M_W_batch, p) for m in microbatches]
peak_stage2 = [peak_memory(2, m, M_B_batch, M_W_batch, p) for m in microbatches]
peak_stage3 = [peak_memory(3, m, M_B_batch, M_W_batch, p) for m in microbatches]

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(microbatches, peak_stage1, label='Stage 1', linewidth=2)
plt.plot(microbatches, peak_stage2, label='Stage 2', linewidth=2)
plt.plot(microbatches, peak_stage3, label='Stage 3', linewidth=2)

# Add legend with memory info
plt.legend(title=f"$M_B^{{batch}}={M_B_batch},\ M_W^{{batch}}={M_W_batch}$", fontsize=15, title_fontsize=16)

plt.title("Peak Memory vs Microbatches (p=3)", fontsize=16)
plt.xlabel("Number of microbatches (m)", fontsize=14)
plt.ylabel("Peak Memory", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("peak_memory_plot_p3.png")
plt.show()
