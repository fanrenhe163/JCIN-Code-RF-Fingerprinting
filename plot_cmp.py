import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import numpy as np

# ===================== 参数和路径设置 =====================

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"],
}
rcParams.update(config)

npy_dir = "./PDT/npy_data_cmp/"

device_list = [0, 3, 7, 11, 15]

# 显示范围，完整为 -30dB 到 20dB，每隔5dB一个点
snr_values = np.arange(-15, 25, 5)

results_file_method1 = os.path.join(
    npy_dir, f"method1_results_devices_{len(device_list)}.pkl"
)
results_file_method2 = os.path.join(
    npy_dir, f"method2_results_devices_{len(device_list)}.pkl"
)
results_file_method3 = os.path.join(
    npy_dir, f"method3_results_devices_{len(device_list)}.pkl"
)
results_file_method4 = os.path.join(
    npy_dir, f"method4_results_devices_{len(device_list)}.pkl"
)

with open(results_file_method1, "rb") as f:
    method1_results = pickle.load(f)

with open(results_file_method2, "rb") as f:
    method2_results = pickle.load(f)

with open(results_file_method3, "rb") as f:
    method3_results = pickle.load(f)

with open(results_file_method4, "rb") as f:
    method4_results = pickle.load(f)

snr_values_sorted = sorted(snr_values)
method1_accuracies = [method1_results.get(snr, None) for snr in snr_values_sorted]
method2_accuracies = [method2_results.get(snr, None) for snr in snr_values_sorted]
method3_accuracies = [method3_results.get(snr, None) for snr in snr_values_sorted]
method4_accuracies = [method4_results.get(snr, None) for snr in snr_values_sorted]

valid_indices = [i for i, acc in enumerate(method1_accuracies) if acc is not None]
snr_values_valid = [snr_values_sorted[i] for i in valid_indices]
method1_accuracies = [method1_accuracies[i] for i in valid_indices]
method2_accuracies = [method2_accuracies[i] for i in valid_indices]
method3_accuracies = [method3_accuracies[i] for i in valid_indices]
method4_accuracies = [method4_accuracies[i] for i in valid_indices]

fig, ax = plt.subplots(figsize=(10, 8))

color_alg1 = "blue"
color_alg2 = "red"
color_alg3 = "green"

ax.plot(
    snr_values_valid,
    method4_accuracies,
    linestyle="--",
    marker="o",
    color=color_alg1,
    linewidth=2,
    label="Algorithm 1 (Full)",
)

ax.plot(
    snr_values_valid,
    method1_accuracies,
    linestyle="-",
    marker="o",
    color=color_alg1,
    linewidth=2,
    label="Algorithm 1 (Partial)",
)

ax.plot(
    snr_values_valid,
    method2_accuracies,
    linestyle="-",
    marker="o",
    color=color_alg2,
    linewidth=2,
    label="Algorithm 2: Wavelet+ReliefF+PCA+SVM",
)

ax.plot(
    snr_values_valid,
    method3_accuracies,
    linestyle="-",
    marker="o",
    color=color_alg3,
    linewidth=2,
    label="Algorithm 3: Wavelet+ReliefF+PCA+RF",
)

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Average Accuracy (%)")
ax.set_title(f"Performance of Different Algorithms")

ax.legend(fontsize=12)
plt.grid(True)

fig_dir = "./figs"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

plt.savefig(
    os.path.join(fig_dir, f"不同算法性能分析.svg"),
    dpi=600,
    facecolor="w",
    edgecolor="w",
    orientation="portrait",
    pad_inches=0.1,
)
plt.savefig(
    os.path.join(fig_dir, f"不同算法性能分析.png"),
    dpi=600,
    facecolor="w",
    edgecolor="w",
    orientation="portrait",
    pad_inches=0.1,
)
plt.close()
