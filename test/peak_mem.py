import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 3  # number of stages
M_B_batch = 50  # batch-level backward memory (arbitrary units)
M_W_batch = 20  # batch-level weight gradient memory (arbitrary units)

# TSPipe peak memory (set this value as needed)
tspipe_peak_memory = 50  # example value

# Function to compute peak memory for given stage i
def peak_memory(i, m, M_B_batch, M_W_batch, p):
    return ((p - i) * M_B_batch + (m - (p - i)) * M_W_batch) / m

# Range of microbatches (start from 6)
microbatches = np.arange(6, 21, 1)  # m = 6 to 20

# Compute peak memory for each stage
peak_stage1 = [peak_memory(1, m, M_B_batch, M_W_batch, p) for m in microbatches]
peak_stage2 = [peak_memory(2, m, M_B_batch, M_W_batch, p) for m in microbatches]
peak_stage3 = [peak_memory(3, m, M_B_batch, M_W_batch, p) for m in microbatches]

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(microbatches, peak_stage1, label='Stage 1', linewidth=2)
plt.plot(microbatches, peak_stage2, label='Stage 2', linewidth=2)
plt.plot(microbatches, peak_stage3, label='Stage 3', linewidth=2)

# Update legend to mention ZBTS_{p-1} and TSPipe
legend_text = (
    f"$M_B^{{batch}}={M_B_batch},\ M_W^{{batch}}={M_W_batch}$\n"
    f"Peak memory $ZBTS_{{p-1}}$=TSPipe$={tspipe_peak_memory}$)"
)
plt.legend(title=legend_text, fontsize=15, title_fontsize=16)

plt.title("Peak Memory vs Microbatches (p=3)", fontsize=16)
plt.xlabel("Number of microbatches (m)", fontsize=14)
plt.ylabel("Peak Memory", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)

# Set integer ticks for x and y axes
plt.xticks(microbatches)
y_min = int(min(min(peak_stage1), min(peak_stage2), min(peak_stage3)))
y_max = int(max(max(peak_stage1), max(peak_stage2), max(peak_stage3))) + 1
plt.yticks(np.arange(y_min, y_max + 1, 5))

plt.savefig("peak_memory_plot_p3.png")
plt.show()
