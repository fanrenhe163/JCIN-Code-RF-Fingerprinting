import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import scipy.io
from scipy.signal import welch, correlate, hilbert
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import pandas as pd
import pickle
import joblib

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================== 参数和路径设置 =====================
num_devices = 17
num_groups = 1000
fs = 76800
# fixed_length = 3000
num_epochs = 500
batch_size = 128
data_dir = "./PDT/data/"
npy_dir = "./PDT/npy_data_cmp/"
model_dir = "./PDT/models_cmp/"
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
feature_set_name = "15_features"
feature_list = [
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
]
device_list = [0, 3, 7, 11, 15]
snr_values = np.arange(-30, 25, 5)
signals_exist = all(
    os.path.exists(os.path.join(npy_dir, f"signals_snr_{snr}_{num_groups}.npy"))
    for snr in snr_values
)
labels_file = os.path.join(npy_dir, f"labels_{num_groups}.npy")
if signals_exist and os.path.exists(labels_file):
    print("从 npy 文件加载信号数据和标签...")
else:
    print("npy 文件不存在，从 mat 文件加载数据并添加噪声后保存为 npy 文件...")
    all_signals = []
    all_labels = []
    for device_num in range(num_devices):
        for group_num in range(1, num_groups + 1):
            file_name = os.path.join(data_dir, f"signal_{group_num}_{device_num}.mat")
            if not os.path.exists(file_name):
                print(f"文件 {file_name} 不存在，跳过该文件。")
                continue
            mat = scipy.io.loadmat(file_name)
            rx_cpfsk_signal = mat["rx_cpfsk_signal"].flatten()
            signal_segment = rx_cpfsk_signal
            all_signals.append(signal_segment)
            all_labels.append(device_num)
    all_signals = np.array(all_signals)
    all_labels = np.array(all_labels)
    np.save(labels_file, all_labels)
    for snr in snr_values:
        print(f"正在处理 SNR = {snr} dB 的数据...")
        noisy_signals = []
        for signal in all_signals:
            signal_power = np.mean(np.abs(signal) ** 2)
            snr_linear = 10 ** (snr / 10)
            noise_power = signal_power / snr_linear
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
            )
            noisy_signal = signal + noise
            noisy_signals.append(noisy_signal)
        noisy_signals = np.array(noisy_signals)
        signals_file = os.path.join(
            npy_dir,
            f"signals_snr_{snr}_{num_groups}.npy",
        )
        np.save(signals_file, noisy_signals)
        print(f"SNR = {snr} dB 的数据已保存为 {signals_file}")
print("所有 SNR 值的数据已准备好。")


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


method4_results = {}  # 记录方法四的结果
for snr in snr_values:
    print(f"\n处理 SNR = {snr} dB 的数据")
    signals_file = os.path.join(
        npy_dir,
        f"signals_snr_{snr}_{num_groups}.npy",
    )
    labels_file = os.path.join(npy_dir, f"labels_{num_groups}.npy")
    all_signals = np.load(signals_file, allow_pickle=True)
    all_labels = np.load(labels_file)
    label_mapping = {device_num: idx for idx, device_num in enumerate(device_list)}
    selected_indices = [i for i, label in enumerate(all_labels) if label in device_list]
    selected_signals = all_signals[selected_indices]
    selected_labels = all_labels[selected_indices]
    selected_labels = np.array([label_mapping[label] for label in selected_labels])
    num_selected_devices = len(device_list)
    print(f"方法四：提取15个特征并使用神经网络分类（不截断信号）")
    features_file = os.path.join(
        npy_dir, f"features_method4_snr_{snr}_devices_{len(device_list)}.npy"
    )
    labels_features_file = os.path.join(
        npy_dir, f"labels_method4_snr_{snr}_devices_{len(device_list)}.npy"
    )
    if os.path.exists(features_file) and os.path.exists(labels_features_file):
        print("特征已提取，正在从文件加载特征和标签...")
        data_method4 = np.load(features_file)
        labels_method4 = np.load(labels_features_file)
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
            third_order_idx = np.argmin(np.abs(fft_freqs - third_order_freq))
            third_order_amplitude = np.abs(fft_signal[third_order_idx])
            fundamental_amplitude = np.abs(fft_signal[fundamental_freq])
            iip3_feature = 10 * np.log10(
                third_order_amplitude / (fundamental_amplitude + 1e-6)
            )
            features["iip3_feature"] = iip3_feature
            # 二阶谐波
            second_order_freq = 2 * fft_freqs[fundamental_freq]
            second_order_idx = np.argmin(np.abs(fft_freqs - second_order_freq))
            second_order_amplitude = np.abs(fft_signal[second_order_idx])
            iip2_feature = 10 * np.log10(
                second_order_amplitude / (fundamental_amplitude + 1e-6)
            )
            features["iip2_feature"] = iip2_feature
            # IQ不平衡特征
            I = np.real(signal_segment)
            Q = np.imag(signal_segment)
            amp_imbalance = 20 * np.log10((np.std(I) + 1e-6) / (np.std(Q) + 1e-6))
            phase_imbalance = np.angle(np.mean(signal_segment))
            features["amp_imbalance"] = amp_imbalance
            features["phase_imbalance"] = phase_imbalance
            # DC偏移特征
            dc_offset_I = np.mean(I)
            dc_offset_Q = np.mean(Q)
            features["dc_offset_I"] = dc_offset_I
            features["dc_offset_Q"] = dc_offset_Q
            ## 功率谱密度特征
            freqs, psd = welch(signal_segment, fs=fs, nperseg=256, return_onesided=True)
            # 频谱质心
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            features["spectral_centroid"] = spectral_centroid
            # 频谱熵
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-6))
            features["spectral_entropy"] = spectral_entropy
            # EVM（误差向量幅度）
            ideal_signal = np.exp(
                1j * 2 * np.pi * fundamental_freq * np.arange(len(signal_segment)) / fs
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
            acpr = 10 * np.log10(np.sum(psd) / (acpr_upper + acpr_lower + 1e-6))
            features["acpr"] = acpr
            # 峭度和偏度
            kurtosis = np.mean(
                (np.real(signal_segment) - np.mean(np.real(signal_segment))) ** 4
            ) / (np.std(np.real(signal_segment)) ** 4 + 1e-6)
            skewness = np.mean(
                (np.real(signal_segment) - np.mean(np.real(signal_segment))) ** 3
            ) / (np.std(np.real(signal_segment)) ** 3 + 1e-6)
            features["kurtosis"] = kurtosis
            features["skewness"] = skewness
            # 瞬时频率特征
            analytic_signal = hilbert(I)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (
                2.0 * np.pi * (1.0 / fs)
            )
            features["mean_instantaneous_frequency"] = np.mean(instantaneous_frequency)
            features["std_instantaneous_frequency"] = np.std(instantaneous_frequency)
            # 自相关特征
            autocorr = correlate(
                np.real(signal_segment), np.real(signal_segment), mode="full"
            )
            autocorr = autocorr[autocorr.size // 2 :]
            autocorr_max = np.max(autocorr)
            features["autocorr_max"] = autocorr_max
            selected_features = [features[feat] for feat in feature_list]
            data.append(selected_features)
        print("特征提取完成。")
        data_method4 = np.array(data)
        labels_method4 = selected_labels
        np.save(features_file, data_method4)
        np.save(labels_features_file, labels_method4)
        print(f"特征已保存到 {features_file}")
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=16)
    input_size = data_method4.shape[1]
    num_classes = num_selected_devices
    all_train_acc = []
    all_test_acc = []
    fold = 0
    for train_index, test_index in skf.split(data_method4, labels_method4):
        fold += 1
        print(f"折数 {fold}:")
        model_save_path_fold = os.path.join(
            model_dir, f"method4_model_snr_{snr}_fold_{fold}.pt"
        )
        if os.path.exists(model_save_path_fold):
            print(f"找到已保存的模型，正在加载 {model_save_path_fold}")
            model = NN(input_size, num_classes).to(device)
            model.load_state_dict(torch.load(model_save_path_fold))
            model.eval()
            with torch.no_grad():
                X_test_fold = data_method4[test_index]
                Y_test_fold = labels_method4[test_index]
                scaler = StandardScaler()
                X_train_fold = data_method4[train_index]
                scaler.fit(X_train_fold)
                X_test_fold = scaler.transform(X_test_fold)
                X_test_fold = torch.tensor(X_test_fold, dtype=torch.float32).to(device)
                Y_test_fold = torch.tensor(Y_test_fold, dtype=torch.long).to(device)
                test_outputs = model(X_test_fold)
                test_preds = torch.argmax(test_outputs, dim=1)
                test_acc = (test_preds == Y_test_fold).float().mean().item()
                print(f"折数 {fold}, 测试准确率: {test_acc:.4f}")
                all_test_acc.append(test_acc)
            continue
        X_train_fold, X_test_fold = data_method4[train_index], data_method4[test_index]
        Y_train_fold, Y_test_fold = (
            labels_method4[train_index],
            labels_method4[test_index],
        )
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32)
        X_test_fold = torch.tensor(X_test_fold, dtype=torch.float32)
        Y_train_fold = torch.tensor(Y_train_fold, dtype=torch.long)
        Y_test_fold = torch.tensor(Y_test_fold, dtype=torch.long)
        model = NN(input_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_dataset = TensorDataset(X_train_fold, Y_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
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
            if (epoch + 1) % 10 == 0:
                train_accuracy = correct / total
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, "
                    f"训练准确率: {train_accuracy:.4f}"
                )
        torch.save(model.state_dict(), model_save_path_fold)
        print(f"模型已保存到 {model_save_path_fold}")
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_fold.to(device))
            train_preds = torch.argmax(train_outputs, dim=1)
            train_acc = (train_preds == Y_train_fold.to(device)).float().mean().item()
            test_outputs = model(X_test_fold.to(device))
            test_preds = torch.argmax(test_outputs, dim=1)
            test_acc = (test_preds == Y_test_fold.to(device)).float().mean().item()
        print(f"折数 {fold}, 最终训练准确率: {train_acc:.4f}")
        print(f"折数 {fold}, 测试准确率: {test_acc:.4f}")
        all_train_acc.append(train_acc)
        all_test_acc.append(test_acc)
    avg_train_acc = np.mean(all_train_acc)
    avg_test_acc = np.mean(all_test_acc)
    print(
        f"方法四 SNR = {snr} dB, 平均训练准确率: {avg_train_acc:.4f}, 平均测试准确率: {avg_test_acc:.4f}"
    )
    method4_results[snr] = avg_test_acc

results_file_method4 = os.path.join(
    npy_dir, f"method4_results_devices_{len(device_list)}.pkl"
)
with open(results_file_method4, "wb") as f:
    pickle.dump(method4_results, f)
print("所有 SNR 值的数据处理完成，结果已保存。")
