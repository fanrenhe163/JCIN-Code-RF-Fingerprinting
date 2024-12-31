import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)
import scipy.io
from scipy.signal import welch, correlate, hilbert
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import pandas as pd
import pickle
from utils import get_device_list_and_log

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================== 参数和路径设置 =====================
snr_values = [-30, 0, 15]
add_snr = True
num_devices = 17
num_groups = 1000
fs = 76800
fixed_length = 2000
num_epochs = 500
batch_size = 128
data_dir = "./PDT/data/"
npy_dir = "./PDT/npy_data_test/"
model_dir = "./PDT/models_test/"
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
feature_sets = {
    "20_features": [
        "iip3_feature",
        "iip2_feature",
        "amp_imbalance",
        "phase_imbalance",
        "dc_offset_I",
        "dc_offset_Q",
        "low_band_power",
        "mid_band_power",
        "high_band_power",
        "spectral_centroid",
        "spectral_entropy",
        "evm_db",
        "acpr",
        "kurtosis",
        "skewness",
        "mean_instantaneous_frequency",
        "std_instantaneous_frequency",
        "envelope_mean",
        "envelope_std",
        "autocorr_max",
    ],
    "15_features": [
        "iip3_feature",
        "iip2_feature",
        "amp_imbalance",
        "phase_imbalance",
        "dc_offset_I",
        "dc_offset_Q",
        "spectral_centroid",
        "spectral_entropy",
        "evm_db",
        "acpr",
        "kurtosis",
        "skewness",
        "mean_instantaneous_frequency",
        "std_instantaneous_frequency",
        "autocorr_max",
    ],
}
# selection列表（bug：第一个selection文本保存为空）
selection_list = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class Logger(object):
    def __init__(self, filename="output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


for SNR in snr_values:
    print(f"\n========== 处理 SNR: {SNR} dB ==========")
    signals_file = os.path.join(npy_dir, f"signals_{num_groups}_snr_{SNR}.npy")
    labels_file = os.path.join(npy_dir, f"labels_{num_groups}_snr_{SNR}.npy")
    if os.path.exists(signals_file) and os.path.exists(labels_file):
        all_signals = np.load(signals_file, allow_pickle=True)
        all_labels = np.load(labels_file)
        print("从 npy 文件加载信号数据和标签...")
    else:
        print("npy 文件不存在，从 mat 文件加载数据并保存为 npy 文件...")
        all_signals = []
        all_labels = []
        for device_num in range(num_devices):
            for group_num in range(1, num_groups + 1):
                file_name = os.path.join(
                    data_dir, f"signal_{group_num}_{device_num}.mat"
                )
                mat = scipy.io.loadmat(file_name)
                rx_cpfsk_signal = mat["rx_cpfsk_signal"].flatten()
                # 提取固定长度的信号段（可选）
                if len(rx_cpfsk_signal) >= fixed_length:
                    # signal_segment = rx_cpfsk_signal[:fixed_length]
                    signal_segment = rx_cpfsk_signal[:]
                else:
                    signal_segment = np.pad(
                        rx_cpfsk_signal,
                        (0, fixed_length - len(rx_cpfsk_signal)),
                        "constant",
                        constant_values=0,
                    )
                if add_snr:
                    signal_power = np.mean(np.abs(signal_segment) ** 2)
                    snr_linear = 10 ** (SNR / 10)
                    noise_power = signal_power / snr_linear
                    noise = np.sqrt(noise_power / 2) * (
                        np.random.randn(len(signal_segment))
                        + 1j * np.random.randn(len(signal_segment))
                    )
                    signal_segment = signal_segment + noise
                all_signals.append(signal_segment)
                all_labels.append(device_num)
        all_signals = np.array(all_signals)
        all_labels = np.array(all_labels)
        np.save(signals_file, all_signals)
        np.save(labels_file, all_labels)
        print("数据已保存为 npy 文件。")

    class NN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.fc4 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            return x

    average_train_accuracies = {}
    average_test_accuracies = {}
    for feature_set_name, feature_list in feature_sets.items():
        print(f"\n========== 处理特征集：{feature_set_name} ==========")
        average_train_accuracies[feature_set_name] = []
        average_test_accuracies[feature_set_name] = []
        for selection in selection_list:
            device_list, img_dir, log_filename = get_device_list_and_log(
                selection, num_devices
            )
            img_dir = os.path.join(img_dir, feature_set_name, f"snr_{SNR}")
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            log_filename = log_filename.replace(
                ".txt", f"_{feature_set_name}_snr_{SNR}.txt"
            )
            original_stdout = sys.stdout
            sys.stdout = Logger(log_filename)
            try:
                print(f"Processing selection {selection}")
                label_mapping = {
                    device_num: idx for idx, device_num in enumerate(device_list)
                }
                selected_indices = [
                    i for i, label in enumerate(all_labels) if label in device_list
                ]
                selected_signals = all_signals[selected_indices]
                selected_labels = all_labels[selected_indices]
                selected_labels = np.array(
                    [label_mapping[label] for label in selected_labels]
                )
                num_selected_devices = len(device_list)
                features_file = os.path.join(
                    npy_dir,
                    f"features_selection_{selection}_{feature_set_name}_snr_{SNR}.npy",
                )
                labels_features_file = os.path.join(
                    npy_dir,
                    f"labels_selection_{selection}_{feature_set_name}_snr_{SNR}.npy",
                )
                if os.path.exists(features_file) and os.path.exists(
                    labels_features_file
                ):
                    print("特征已提取，正在从文件加载特征和标签...")
                    data = np.load(features_file)
                    labels = np.load(labels_features_file)
                else:
                    print("开始特征提取...")
                    data = []
                    for i in range(len(selected_signals)):
                        signal_segment = selected_signals[i]
                        device_num = selected_labels[i]
                        features = {}
                        ## IIP3 相关特征
                        fft_signal = np.fft.fft(signal_segment)
                        fft_freqs = np.fft.fftfreq(len(signal_segment), 1 / fs)
                        fundamental_freq = np.argmax(np.abs(fft_signal))
                        # 三阶谐波
                        third_order_freq = 3 * fft_freqs[fundamental_freq]
                        third_order_idx = np.argmin(
                            np.abs(fft_freqs - third_order_freq)
                        )
                        third_order_amplitude = np.abs(fft_signal[third_order_idx])
                        fundamental_amplitude = np.abs(fft_signal[fundamental_freq])
                        iip3_feature = 10 * np.log10(
                            third_order_amplitude / (fundamental_amplitude + 1e-6)
                        )
                        features["iip3_feature"] = iip3_feature
                        # 二阶谐波
                        second_order_freq = 2 * fft_freqs[fundamental_freq]
                        second_order_idx = np.argmin(
                            np.abs(fft_freqs - second_order_freq)
                        )
                        second_order_amplitude = np.abs(fft_signal[second_order_idx])
                        iip2_feature = 10 * np.log10(
                            second_order_amplitude / (fundamental_amplitude + 1e-6)
                        )
                        features["iip2_feature"] = iip2_feature
                        # IQ不平衡特征
                        I = np.real(signal_segment)
                        Q = np.imag(signal_segment)
                        amp_imbalance = 20 * np.log10(
                            (np.std(I) + 1e-6) / (np.std(Q) + 1e-6)
                        )
                        phase_imbalance = np.angle(np.mean(signal_segment))
                        features["amp_imbalance"] = amp_imbalance
                        features["phase_imbalance"] = phase_imbalance
                        # DC偏移特征
                        dc_offset_I = np.mean(I)
                        dc_offset_Q = np.mean(Q)
                        features["dc_offset_I"] = dc_offset_I
                        features["dc_offset_Q"] = dc_offset_Q
                        ## 功率谱密度特征
                        freqs, psd = welch(
                            signal_segment, fs=fs, nperseg=256, return_onesided=True
                        )
                        # 带通功率
                        low_band_power = np.sum(psd[(freqs >= 0) & (freqs <= 1000)])
                        mid_band_power = np.sum(psd[(freqs > 1000) & (freqs <= 2000)])
                        high_band_power = np.sum(psd[(freqs > 2000) & (freqs <= 3000)])
                        features["low_band_power"] = low_band_power
                        features["mid_band_power"] = mid_band_power
                        features["high_band_power"] = high_band_power
                        # 频谱质心
                        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                        features["spectral_centroid"] = spectral_centroid
                        # 频谱熵
                        psd_norm = psd / np.sum(psd)
                        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-6))
                        features["spectral_entropy"] = spectral_entropy
                        # EVM（误差向量幅度）
                        ideal_signal = np.exp(
                            1j
                            * 2
                            * np.pi
                            * fundamental_freq
                            * np.arange(len(signal_segment))
                            / fs
                        )
                        evm = np.sqrt(
                            np.mean(np.abs(signal_segment - ideal_signal) ** 2)
                        ) / np.sqrt(np.mean(np.abs(ideal_signal) ** 2))
                        evm_db = 20 * np.log10(evm)
                        features["evm_db"] = evm_db
                        # ACPR（邻道功率比）
                        acpr_upper = np.sum(
                            psd[
                                np.where(
                                    (freqs > fundamental_freq)
                                    & (freqs <= fundamental_freq + fs / 2)
                                )
                            ]
                        )
                        acpr_lower = np.sum(
                            psd[
                                np.where(
                                    (freqs > fundamental_freq - fs / 2)
                                    & (freqs <= fundamental_freq)
                                )
                            ]
                        )
                        acpr = 10 * np.log10(
                            np.sum(psd) / (acpr_upper + acpr_lower + 1e-6)
                        )
                        features["acpr"] = acpr
                        # 峭度和偏度
                        kurtosis = np.mean(
                            (np.real(signal_segment) - np.mean(np.real(signal_segment)))
                            ** 4
                        ) / (np.std(np.real(signal_segment)) ** 4 + 1e-6)
                        skewness = np.mean(
                            (np.real(signal_segment) - np.mean(np.real(signal_segment)))
                            ** 3
                        ) / (np.std(np.real(signal_segment)) ** 3 + 1e-6)
                        features["kurtosis"] = kurtosis
                        features["skewness"] = skewness
                        # 瞬时频率特征
                        analytic_signal = hilbert(np.real(signal_segment))
                        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                        instantaneous_frequency = np.diff(instantaneous_phase) / (
                            2.0 * np.pi * (1.0 / fs)
                        )
                        features["mean_instantaneous_frequency"] = np.mean(
                            instantaneous_frequency
                        )
                        features["std_instantaneous_frequency"] = np.std(
                            instantaneous_frequency
                        )
                        # 信号包络特征
                        envelope = np.abs(analytic_signal)
                        envelope_mean = np.mean(envelope)
                        envelope_std = np.std(envelope)
                        features["envelope_mean"] = envelope_mean
                        features["envelope_std"] = envelope_std
                        # 自相关特征
                        autocorr = correlate(
                            np.real(signal_segment),
                            np.real(signal_segment),
                            mode="full",
                        )
                        autocorr = autocorr[autocorr.size // 2 :]
                        autocorr_max = np.max(autocorr)
                        features["autocorr_max"] = autocorr_max
                        selected_features = [features[feat] for feat in feature_list]
                        data.append(selected_features)
                    print("特征提取完成。")
                    data = np.array(data)
                    labels = selected_labels
                    np.save(features_file, data)
                    np.save(labels_features_file, labels)
                    print(f"特征已保存到 {features_file}")
                    data_df = pd.DataFrame(data, columns=feature_list)
                    corr_matrix = data_df.corr()
                    np.savez(
                        os.path.join(
                            img_dir,
                            f"corr_data_selection_{selection}_{feature_set_name}_snr_{SNR}.npz",
                        ),
                        corr_matrix=corr_matrix.values,
                        feature_names=feature_list,
                    )
                k = 5
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=16)
                input_size = data.shape[1]
                num_classes = num_selected_devices
                all_train_acc = []
                all_test_acc = []
                all_precision = []
                all_recall = []
                all_f1 = []
                all_auc = []
                all_fpr = []
                all_tpr = []
                all_roc_auc = []
                all_fold_numbers = []
                all_confusion_matrices = []
                fold = 0
                for train_index, test_index in skf.split(data, labels):
                    fold += 1
                    print(f"\n===== 第 {fold} 折训练开始 =====")
                    X_train_fold, X_test_fold = data[train_index], data[test_index]
                    Y_train_fold, Y_test_fold = labels[train_index], labels[test_index]
                    scaler = StandardScaler()
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    X_test_fold = scaler.transform(X_test_fold)
                    X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32)
                    X_test_fold = torch.tensor(X_test_fold, dtype=torch.float32)
                    Y_train_fold = torch.tensor(Y_train_fold, dtype=torch.long)
                    Y_test_fold = torch.tensor(Y_test_fold, dtype=torch.long)
                    model_file = os.path.join(
                        model_dir,
                        f"model_selection_{selection}_{feature_set_name}_snr_{SNR}_fold_{fold}.pth",
                    )
                    if os.path.exists(model_file):
                        print(f"加载已保存的模型：{model_file}")
                        model = NN(input_size, num_classes).to(device)
                        model.load_state_dict(torch.load(model_file))
                    else:
                        model = NN(input_size, num_classes).to(device)
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=0.001)
                        train_dataset = TensorDataset(X_train_fold, Y_train_fold)
                        train_loader = DataLoader(
                            train_dataset, batch_size=batch_size, shuffle=True
                        )
                        epoch_num = 0
                        for epoch in range(num_epochs):
                            model.train()
                            running_loss = 0.0
                            correct = 0
                            total = 0
                            for i, (inputs, targets) in enumerate(train_loader):
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                running_loss += loss.item()
                                _, predicted = torch.max(outputs.data, 1)
                                total += targets.size(0)
                                correct += (predicted == targets).sum().item()
                            train_accuracy = 100 * correct / total
                            epoch_num += 1
                            if epoch_num % 10 == 0:
                                print(
                                    f"折数 {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, "
                                    f"训练准确率: {train_accuracy:.2f}%"
                                )
                        torch.save(model.state_dict(), model_file)
                        print(f"模型已保存到 {model_file}")
                    model.eval()
                    with torch.no_grad():
                        X_train_fold = X_train_fold.to(device)
                        X_test_fold = X_test_fold.to(device)
                        Y_train_fold = Y_train_fold.to(device)
                        Y_test_fold = Y_test_fold.to(device)
                        train_outputs = model(X_train_fold)
                        train_preds = torch.argmax(train_outputs, dim=1)
                        train_acc = (train_preds == Y_train_fold).float().mean().item()
                        test_outputs = model(X_test_fold)
                        test_preds = torch.argmax(test_outputs, dim=1)
                        test_acc = (test_preds == Y_test_fold).float().mean().item()
                    print(f"折数 {fold}, 最终训练准确率: {train_acc:.4f}")
                    print(f"折数 {fold}, 测试准确率: {test_acc:.4f}")
                    all_train_acc.append(train_acc)
                    all_test_acc.append(test_acc)
                    Y_test_np = Y_test_fold.cpu().numpy()
                    test_preds_np = test_preds.cpu().numpy()
                    precision = precision_score(
                        Y_test_np, test_preds_np, average="macro"
                    )
                    recall = recall_score(Y_test_np, test_preds_np, average="macro")
                    f1 = f1_score(Y_test_np, test_preds_np, average="macro")
                    print(f"折数 {fold}, 精确率: {precision:.4f}")
                    print(f"折数 {fold}, 召回率: {recall:.4f}")
                    print(f"折数 {fold}, F1分数: {f1:.4f}")
                    all_precision.append(precision)
                    all_recall.append(recall)
                    all_f1.append(f1)
                    test_probs = torch.softmax(test_outputs, dim=1)
                    test_probs_np = test_probs.cpu().numpy()
                    Y_test_binarized = label_binarize(
                        Y_test_np, classes=np.arange(num_classes)
                    )
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(num_classes):
                        fpr[i], tpr[i], _ = roc_curve(
                            Y_test_binarized[:, i], test_probs_np[:, i]
                        )
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    fpr["micro"], tpr["micro"], _ = roc_curve(
                        Y_test_binarized.ravel(), test_probs_np.ravel()
                    )
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    print(f"折数 {fold}, AUC（micro平均）: {roc_auc['micro']:.4f}")
                    all_auc.append(roc_auc["micro"])
                    all_fpr.append(fpr["micro"])
                    all_tpr.append(tpr["micro"])
                    all_roc_auc.append(roc_auc["micro"])
                    all_fold_numbers.append(fold)
                    conf_matrix = confusion_matrix(Y_test_np, test_preds_np)
                    all_confusion_matrices.append(conf_matrix)
                avg_conf_matrix = sum(all_confusion_matrices) / k
                np.save(
                    os.path.join(
                        img_dir,
                        f"confusion_matrix_{selection}_{feature_set_name}_snr_{SNR}.npy",
                    ),
                    avg_conf_matrix,
                )
                roc_data = {
                    "all_fpr": all_fpr,
                    "all_tpr": all_tpr,
                    "all_roc_auc": all_roc_auc,
                    "all_fold_numbers": all_fold_numbers,
                }
                with open(
                    os.path.join(
                        img_dir,
                        f"roc_data_selection_{selection}_{feature_set_name}_snr_{SNR}.pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(roc_data, f)
                print("\n===== 最终平均结果 =====")
                print(f"平均训练准确率: {np.mean(all_train_acc):.4f}")
                print(f"平均测试准确率: {np.mean(all_test_acc):.4f}")
                print(f"平均精确率: {np.mean(all_precision):.4f}")
                print(f"平均召回率: {np.mean(all_recall):.4f}")
                print(f"平均F1分数: {np.mean(all_f1):.4f}")
                print(f"平均AUC值: {np.mean(all_auc):.4f}")
                average_train_accuracies[feature_set_name].append(
                    np.mean(all_train_acc)
                )
                average_test_accuracies[feature_set_name].append(np.mean(all_test_acc))
            finally:
                sys.stdout = original_stdout
        np.savez(
            os.path.join(
                npy_dir,
                f"average_accuracies_{feature_set_name}_{num_groups}_snr_{SNR}.npz",
            ),
            selection_list=selection_list,
            average_train_accuracies=average_train_accuracies[feature_set_name],
            average_test_accuracies=average_test_accuracies[feature_set_name],
        )
    params = {
        "num_devices": num_devices,
        "num_groups": num_groups,
        "feature_sets": feature_sets,
        "selection_list": selection_list,
        "snr_values": snr_values,
    }
    with open(os.path.join(npy_dir, "params_sync.npy"), "wb") as f:
        np.save(f, params)
