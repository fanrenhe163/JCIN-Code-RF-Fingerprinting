import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pickle

npy_dir = "./PDT/npy_data_test/"

with open(os.path.join(npy_dir, "params_sync.npy"), "rb") as f:
    params = np.load(f, allow_pickle=True).item()

num_devices = params["num_devices"]
num_groups = params["num_groups"]
feature_sets = params["feature_sets"]
selection_list = params["selection_list"]
snr_values = params["snr_values"]

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"],
}
rcParams.update(config)

from utils import get_device_list_and_log

figs_dir = "./figs"
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

available_colors = ["b", "r", "g", "c", "m", "y", "k"]
available_line_styles = ["-", "--", "-.", ":"]
feature_set_names = list(feature_sets.keys())

colors = {}
for idx, feature_set_name in enumerate(feature_set_names):
    colors[feature_set_name] = available_colors[idx % len(available_colors)]

line_styles = {"train": "--", "test": "-"}
markers = {"train": "o", "test": "s"}

feature_name_to_latex = {
    "iip3_feature": r"$F_{H_3}$",
    "iip2_feature": r"$F_{H_2}$",
    "amp_imbalance": r"$F_A$",
    "phase_imbalance": r"$F_P$",
    "dc_offset_I": r"$F_{\text{DCI}}$",
    "dc_offset_Q": r"$F_{\text{DCQ}}$",
    "low_band_power": r"$F_{\text{LP}}$",
    "mid_band_power": r"$F_{\text{MP}}$",
    "high_band_power": r"$F_{\text{HP}}$",
    "spectral_centroid": r"$F_{\text{SC}}$",
    "spectral_entropy": r"$F_{\text{SE}}$",
    "evm_db": r"$F_{\text{EVM}}$",
    "acpr": r"$F_{\text{ACPR}}$",
    "kurtosis": r"$F_K$",
    "skewness": r"$F_S$",
    "mean_instantaneous_frequency": r"$F_{\text{MIF}}$",
    "std_instantaneous_frequency": r"$F_{\text{SIF}}$",
    "envelope_mean": r"$F_{\text{EM}}$",
    "envelope_std": r"$F_{\text{ES}}$",
    "autocorr_max": r"$F_{\text{ACmax}}$",
}

plt.rcParams["text.usetex"] = False

for SNR in snr_values:
    figs_dir_snr = os.path.join(figs_dir, f"snr_{SNR}")
    if not os.path.exists(figs_dir_snr):
        os.makedirs(figs_dir_snr)

    for selection in selection_list:
        device_list, img_dir, log_filename = get_device_list_and_log(
            selection, num_devices
        )
        for feature_set_name, feature_list in feature_sets.items():
            img_dir_feature = os.path.join(img_dir, feature_set_name, f"snr_{SNR}")
            figs_subdir = os.path.join(
                figs_dir_snr, feature_set_name, f"selection_{selection}"
            )

            if not os.path.exists(figs_subdir):
                os.makedirs(figs_subdir)

            feature_latex_list = [
                feature_name_to_latex.get(name, name) for name in feature_list
            ]

            corr_data_file = os.path.join(
                img_dir_feature,
                f"corr_data_selection_{selection}_{feature_set_name}_snr_{SNR}.npz",
            )

            if os.path.exists(corr_data_file):
                corr_data = np.load(corr_data_file, allow_pickle=True)
                corr_matrix = corr_data["corr_matrix"]
                num_features = corr_matrix.shape[0]

                plt.figure(figsize=(10, 8))
                pltsns = sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    linewidths=0.5,
                    xticklabels=range(1, num_features + 1),
                    # yticklabels=range(1, num_features + 1),
                    # xticklabels=feature_latex_list,
                    yticklabels=feature_latex_list,
                )
                for snslabel in pltsns.get_yticklabels():
                    snslabel.set_horizontalalignment("left")
                    snslabel.set_x(-0.06)
                    snslabel.set_rotation(0)
                plt.title(f"Feature Correlation Heatmap")
                plt.tight_layout()

                svgfile = os.path.join(figs_subdir, f"特征相关热力图.svg")
                pngfile = os.path.join(figs_subdir, f"特征相关热力图.png")
                plt.savefig(
                    svgfile,
                    dpi=600,
                    facecolor="w",
                    edgecolor="w",
                    orientation="portrait",
                    pad_inches=0.1,
                )
                plt.savefig(
                    pngfile,
                    dpi=600,
                    facecolor="w",
                    edgecolor="w",
                    orientation="portrait",
                    pad_inches=0.1,
                )
                plt.close()
            else:
                print(
                    f"未找到 selection {selection}, feature set {feature_set_name}, SNR {SNR} 的相关性数据文件。"
                )

            roc_data_file = os.path.join(
                img_dir_feature,
                f"roc_data_selection_{selection}_{feature_set_name}_snr_{SNR}.pkl",
            )

            if os.path.exists(roc_data_file):
                with open(roc_data_file, "rb") as f:
                    roc_data = pickle.load(f)
                all_fpr = roc_data["all_fpr"]
                all_tpr = roc_data["all_tpr"]
                all_roc_auc = roc_data["all_roc_auc"]
                all_fold_numbers = roc_data["all_fold_numbers"]

                num_folds = len(all_fold_numbers)

                for i in range(num_folds):
                    fpr = all_fpr[i]
                    tpr = all_tpr[i]
                    roc_auc_value = all_roc_auc[i]
                    fold_number = all_fold_numbers[i]

                    plt.figure(figsize=(10, 8))
                    plt.plot(
                        fpr,
                        tpr,
                        linewidth=2,
                        label=f"Fold {fold_number} ROC Curve (AUC = {roc_auc_value:.2f})",
                    )
                    plt.xlabel("False Acceptance Rate (FAR)")
                    plt.ylabel("Recall")
                    plt.title(f"ROC Curve")
                    plt.legend(loc="lower right")

                    svgfile = os.path.join(
                        figs_subdir, f"ROC曲线 第{fold_number}折.svg"
                    )
                    pngfile = os.path.join(
                        figs_subdir, f"ROC曲线 第{fold_number}折.png"
                    )
                    plt.savefig(
                        svgfile,
                        dpi=600,
                        facecolor="w",
                        edgecolor="w",
                        orientation="portrait",
                        pad_inches=0.1,
                    )
                    plt.savefig(
                        pngfile,
                        dpi=600,
                        facecolor="w",
                        edgecolor="w",
                        orientation="portrait",
                        pad_inches=0.1,
                    )
                    plt.close()
            else:
                print(
                    f"未找到 selection {selection}, feature set {feature_set_name}, SNR {SNR} 的 ROC 数据文件。"
                )

            conf_matrix_file = os.path.join(
                img_dir_feature,
                f"confusion_matrix_{selection}_{feature_set_name}_snr_{SNR}.npy",
            )

            if os.path.exists(conf_matrix_file):
                conf_matrix = np.load(conf_matrix_file)
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    conf_matrix, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5
                )
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(f"Confusion Matrix")
                plt.tight_layout()

                svgfile = os.path.join(figs_subdir, f"混淆矩阵.svg")
                pngfile = os.path.join(figs_subdir, f"混淆矩阵.png")
                plt.savefig(
                    svgfile,
                    dpi=600,
                    facecolor="w",
                    edgecolor="w",
                    orientation="portrait",
                    pad_inches=0.1,
                )
                plt.savefig(
                    pngfile,
                    dpi=600,
                    facecolor="w",
                    edgecolor="w",
                    orientation="portrait",
                    pad_inches=0.1,
                )
                plt.close()
            else:
                print(
                    f"未找到 selection {selection}, feature set {feature_set_name}, SNR {SNR} 的混淆矩阵文件。"
                )
    colors = {
        "20_features_train": "#1F78B4",  # 20特征集，训练集，深蓝色
        "20_features_test": "#A6CEE3",  # 20特征集，测试集，浅蓝色
        "15_features_train": "#E31A1C",  # 15特征集，训练集，深红色
        "15_features_test": "#FB9A99",  # 15特征集，测试集，浅红色
    }
    hatch_patterns = {"train": "", "test": "//"}
    bar_width = 0.2
    plt.figure(figsize=(10, 8))
    num_selections = len(selection_list)
    indices = np.arange(num_selections)
    total_width = bar_width * len(feature_sets) * 2
    offset = -total_width / 2 + bar_width / 2
    for feature_set_idx, feature_set_name in enumerate(feature_sets.keys()):
        average_accuracies_file = os.path.join(
            npy_dir, f"average_accuracies_{feature_set_name}_{num_groups}_snr_{SNR}.npz"
        )
        if os.path.exists(average_accuracies_file):
            avg_data = np.load(average_accuracies_file)
            selection_list = avg_data["selection_list"]
            average_train_accuracies = avg_data["average_train_accuracies"] * 100
            average_test_accuracies = avg_data["average_test_accuracies"] * 100
            selection_list = selection_list[1:]
            average_train_accuracies = average_train_accuracies[1:]
            average_test_accuracies = average_test_accuracies[1:]
            num_selections = len(selection_list)
            indices = np.arange(num_selections)
            train_positions = indices + offset
            test_positions = indices + offset + bar_width
            plt.bar(
                train_positions,
                average_train_accuracies,
                bar_width,
                color=colors[f"{feature_set_name}_train"],
                edgecolor="black",
                label=f"Training ({feature_set_name})",
            )
            plt.bar(
                test_positions,
                average_test_accuracies,
                bar_width,
                color=colors[f"{feature_set_name}_test"],
                edgecolor="black",
                hatch=hatch_patterns["test"],
                label=f"Testing ({feature_set_name})",
            )
            offset += bar_width * 2
        else:
            print(
                f"未找到 feature set {feature_set_name}, SNR {SNR} 的平均准确率文件。"
            )
    plt.xlabel("Device Set")
    plt.ylabel("Average Accuracy (%)")
    plt.title(f"Average Training and Test Accuracy (SNR = {SNR} dB)")
    plt.xticks(indices, selection_list)
    if SNR == 15:
        plt.ylim(80, 102)
    if SNR == 0:
        plt.ylim(60, 102)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    svgfile = os.path.join(figs_dir_snr, f"平均训练和测试准确率柱状图.svg")
    pngfile = os.path.join(figs_dir_snr, f"平均训练和测试准确率柱状图.png")
    plt.savefig(
        svgfile,
        dpi=600,
        facecolor="w",
        edgecolor="w",
        orientation="portrait",
        pad_inches=0.1,
    )
    plt.savefig(
        pngfile,
        dpi=600,
        facecolor="w",
        edgecolor="w",
        orientation="portrait",
        pad_inches=0.1,
    )
    plt.close()
