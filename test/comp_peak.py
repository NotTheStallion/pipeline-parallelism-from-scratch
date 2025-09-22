import numpy as np

import matplotlib.pyplot as plt

# Parameters
M_B_batch = 50  # batch-level backward memory (arbitrary units)
M_W_batch = 20  # batch-level weight gradient memory (arbitrary units)

# TSPipe peak memory (set this value as needed)
tspipe_peak_memory = 50  # example value

# Function to compute peak memory for given stage i
def peak_memory(i, m, M_B_batch, M_W_batch, p):
    return ((p - i) * M_B_batch + (m - (p - i)) * M_W_batch) / m


p_values = np.arange(2, 21, 2)
microbatches_values = [np.arange(2*p-1, 51, 1) for p in p_values]

plt.figure(figsize=(10, 5))
for p, microbatches in zip(p_values, microbatches_values):
    # Memory of GPU 1 (highest) from p total GPUs.
    peak_gpu1 = [peak_memory(1, m, M_B_batch, M_W_batch, p) for m in microbatches]
    
    plt.plot(microbatches, peak_gpu1, label=f"{p} GPUs (GPU 1)", linewidth=2)


# Plotting

# Update legend to mention ZBTS_{p-1} and TSPipe
legend_text = (
    f"$M_B^{{batch}}={M_B_batch},\ M_W^{{batch}}={M_W_batch}$\n"
    f"Peak memory $ZBTS_{{p-1}}$=TSPipe$={tspipe_peak_memory}$)"
)
plt.legend(title=legend_text, fontsize=12, title_fontsize=14)

plt.title("Peak Memory of GPU 1 vs Microbatches (p=3)", fontsize=16)
plt.xlabel("Number of microbatches (m)", fontsize=14)
plt.ylabel("Peak Memory (GPU 1)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)

# Set integer ticks for x and y axes
plt.xticks(microbatches)
y_min = int(min(peak_gpu1))
y_max = int(max(peak_gpu1)) + 1
plt.yticks(np.arange(y_min, y_max + 1, 5))

plt.savefig("peak_memory_gpu1_plot_p3.png")
plt.show()