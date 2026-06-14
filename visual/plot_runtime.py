import os
import matplotlib.pyplot as plt
import numpy as np

# Set style parameters for a clean, modern look
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linewidth'] = 0.5

# Data definition
# 1. AMD EPYC 9754 (128 Cores, 256 Threads)
epyc_threads = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
epyc_runtimes = np.array([2955.0891, 1601.7501, 809.0611, 405.4009, 218.6517, 131.8581, 79.5317, 48.6835, 27.1182])
epyc_speedups = epyc_runtimes[0] / epyc_runtimes

# 2. Intel Core i7-10700K (8 Cores, 16 Threads)
i7_threads = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
i7_runtimes = np.array([3780.9796, 1918.7933, 1328.3810, 1036.3911, 866.1966, 748.9263, 660.6184, 594.8885, 
                        599.1490, 551.8462, 507.1294, 478.7060, 452.9379, 425.4405, 401.5139, 385.6999])
i7_speedups = i7_runtimes[0] / i7_runtimes

# Create output folder
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)

# ----------------- PLOT 1: AMD EPYC 9754 Benchmark -----------------
fig_epyc, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(14, 6))

# Left Subplot: Execution Time
ax_time.plot(epyc_threads, epyc_runtimes, marker='o', markersize=8, color='#0F4C81', linewidth=2.5, label='Actual Runtime')
for x, y in zip(epyc_threads, epyc_runtimes):
    label = f"{y:.1f}s"
    # Position adjustments for clarity
    xytext = (0, 8) if x != 256 else (15, -10)
    ax_time.annotate(label, (x, y), textcoords="offset points", xytext=xytext, ha='center', fontsize=9, color='#333333')

ax_time.set_xscale('log', base=2)
ax_time.set_xticks(epyc_threads)
ax_time.set_xticklabels([str(t) for t in epyc_threads])
ax_time.set_title("Execution Time vs. Thread Count (AMD EPYC 9754)", fontsize=12, fontweight='bold', pad=15)
ax_time.set_xlabel("Number of Threads (log2 scale)", fontsize=10, labelpad=10)
ax_time.set_ylabel("Execution Time (Seconds)", fontsize=10, labelpad=10)
ax_time.grid(True, which="both", linestyle='--')

# Right Subplot: Speedup
ax_speedup.plot(epyc_threads, epyc_speedups, marker='s', markersize=8, color='#E056FD', linewidth=2.5, label='Actual Speedup')
ax_speedup.plot(epyc_threads, epyc_threads, linestyle='--', color='#999999', linewidth=1.5, label='Ideal Speedup')
for x, y in zip(epyc_threads, epyc_speedups):
    label = f"{y:.1f}x"
    ax_speedup.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, color='#333333')

ax_speedup.set_xscale('log', base=2)
ax_speedup.set_xticks(epyc_threads)
ax_speedup.set_xticklabels([str(t) for t in epyc_threads])
ax_speedup.set_title("Speedup Factor vs. Thread Count (AMD EPYC 9754)", fontsize=12, fontweight='bold', pad=15)
ax_speedup.set_xlabel("Number of Threads (log2 scale)", fontsize=10, labelpad=10)
ax_speedup.set_ylabel("Speedup Factor ($T_1 / T_p$)", fontsize=10, labelpad=10)
ax_speedup.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='#EEEEEE')
ax_speedup.grid(True, which="both", linestyle='--')

plt.tight_layout()
epyc_path = os.path.join(output_dir, "epyc_9754_benchmark.png")
fig_epyc.savefig(epyc_path, dpi=300)
plt.close(fig_epyc)
print(f"Saved EPYC benchmark plot to: {epyc_path}")

# ----------------- PLOT 2: Intel Core i7-10700K Benchmark -----------------
fig_i7, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(14, 6))

# Left Subplot: Execution Time
ax_time.plot(i7_threads, i7_runtimes, marker='o', markersize=7, color='#1F77B4', linewidth=2.5, label='Actual Runtime')
for x, y in zip(i7_threads, i7_runtimes):
    if x in [1, 2, 4, 8, 12, 16]:  # Annotate selected points to avoid crowding
        label = f"{y:.1f}s"
        ax_time.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, color='#333333')

ax_time.set_xticks(i7_threads)
ax_time.set_title("Execution Time vs. Thread Count (Intel i7-10700K)", fontsize=12, fontweight='bold', pad=15)
ax_time.set_xlabel("Number of Threads", fontsize=10, labelpad=10)
ax_time.set_ylabel("Execution Time (Seconds)", fontsize=10, labelpad=10)
ax_time.grid(True, linestyle='--')

# Right Subplot: Speedup
ax_speedup.plot(i7_threads, i7_speedups, marker='s', markersize=7, color='#D95F02', linewidth=2.5, label='Actual Speedup')
ax_speedup.plot(i7_threads, i7_threads, linestyle='--', color='#999999', linewidth=1.5, label='Ideal Speedup')
for x, y in zip(i7_threads, i7_speedups):
    if x in [1, 2, 4, 8, 12, 16]:  # Annotate selected points
        label = f"{y:.1f}x"
        ax_speedup.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, color='#333333')

ax_speedup.set_xticks(i7_threads)
ax_speedup.set_title("Speedup Factor vs. Thread Count (Intel i7-10700K)", fontsize=12, fontweight='bold', pad=15)
ax_speedup.set_xlabel("Number of Threads", fontsize=10, labelpad=10)
ax_speedup.set_ylabel("Speedup Factor ($T_1 / T_p$)", fontsize=10, labelpad=10)
ax_speedup.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='#EEEEEE')
ax_speedup.grid(True, linestyle='--')

plt.tight_layout()
i7_path = os.path.join(output_dir, "i7_10700k_benchmark.png")
fig_i7.savefig(i7_path, dpi=300)
plt.close(fig_i7)
print(f"Saved i7 benchmark plot to: {i7_path}")
