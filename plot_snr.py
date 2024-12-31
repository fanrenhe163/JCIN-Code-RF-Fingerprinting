import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import matplotlib.lines as mlines

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"],
}
rcParams.update(config)

npy_dir = "./PDT/npy_data_snr/"
selection_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.tab10.colors

markers = ["o", "s", "v", "^", "<", ">", "d", "p", "*", "h"]

selection_descriptions = {
    0: "3 devices",
    1: "17 devices",
    2: "9 devices, IIP3 (Set 1)",
    3: "9 devices, IIP3 (Set 2)",
    4: "9 devices, IQ imbal (Set 1)",
    5: "9 devices, IQ imbal (Set 2)",
    6: "9 devices, DC offset (Set 1)",
    7: "9 devices, DC offset (Set 2)",
    8: "9 devices, Phase offset (Set 1)",
    9: "9 devices, Phase offset (Set 2)",
}

for i, selection in enumerate(selection_list):
    results_file = os.path.join(npy_dir, f"snr_results_selection_{selection}.pkl")
    with open(results_file, "rb") as f:
        selection_results = pickle.load(f)

    snr_values = sorted(selection_results.keys())
    train_accuracies = [selection_results[snr]["train_accuracy"] for snr in snr_values]
    test_accuracies = [selection_results[snr]["test_accuracy"] for snr in snr_values]

    ax.plot(
        snr_values,
        train_accuracies,
        linestyle="--",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
        label=f"{selection_descriptions[selection]}",
    )
    ax.plot(
        snr_values,
        test_accuracies,
        linestyle="-",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
    )

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Average Accuracy (%)")
ax.set_title("Average Training and Test Accuracy")

train_line = mlines.Line2D([], [], color="black", linestyle="--", label="Training Set")
test_line = mlines.Line2D([], [], color="black", linestyle="-", label="Test Set")

fig.legend(
    handles=[train_line, test_line],
    loc="upper left",
    bbox_to_anchor=(0.13, 0.83),
    fontsize=12,
    frameon=False,
)
ax.legend(fontsize=12)

plt.grid(True)
plt.savefig(
    "./figs/不同设备集下平均准确率和 SNR 的关系.svg",
    dpi=600,
    facecolor="w",
    edgecolor="w",
    orientation="portrait",
    pad_inches=0.1,
)
plt.savefig(
    "./figs/不同设备集下平均准确率和 SNR 的关系.png",
    dpi=600,
    facecolor="w",
    edgecolor="w",
    orientation="portrait",
    pad_inches=0.1,
)
plt.close()
