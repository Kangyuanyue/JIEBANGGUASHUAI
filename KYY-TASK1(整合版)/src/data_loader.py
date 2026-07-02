#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块
负责JSONL解析、音频读取、批量加载与数据增强
"""

import json
import os
import numpy as np
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件，返回字典列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    加载单个音频文件
    Args:
        audio_path: 音频文件路径
        sr: 目标采样率
    Returns:
        (音频信号numpy数组, 采样率)
    """
    try:
        waveform, orig_sr = sf.read(audio_path, dtype='float32')
        # 如果是多声道，取第一声道
        if waveform.ndim > 1:
            waveform = waveform[:, 0]
        # 重采样（如果需要）
        if orig_sr != sr:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=sr)
        return waveform, sr
    except Exception as e:
        print(f"[WARNING] 无法加载音频 {audio_path}: {e}")
        return np.zeros(sr, dtype=np.float32), sr  # 返回1秒静音


def load_audio_batch(audio_paths: List[str], sr: int = 16000,
                     num_workers: int = 4) -> List[Tuple[np.ndarray, int]]:
    """
    批量并行加载音频文件
    Args:
        audio_paths: 音频文件路径列表
        sr: 目标采样率
        num_workers: 并行线程数
    Returns:
        [(音频信号, 采样率), ...]
    """
    results = [None] * len(audio_paths)

    def _load(idx, path):
        return idx, load_audio(path, sr)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_load, i, p) for i, p in enumerate(audio_paths)]
        for future in as_completed(futures):
            idx, audio_data = future.result()
            results[idx] = audio_data

    return results


def load_dataset(jsonl_path: str, base_dir: str) -> List[Dict]:
    """
    加载完整数据集，解析JSONL并构建完整路径
    Args:
        jsonl_path: JSONL文件路径
        base_dir: 数据集根目录（音频相对路径的基准目录）
    Returns:
        数据列表，每项包含:
        {
            'id': int,
            'kws_path': str,      # 唤醒音频完整路径
            'kws_text': str,      # 唤醒文本
            'cmd_path': str,      # 识别音频完整路径
            'label': str or None  # 识别文本标签
        }
    """
    raw_data = load_jsonl(jsonl_path)
    dataset = []
    for item in raw_data:
        entry = {
            'id': item['id'],
            'kws_path': os.path.join(base_dir, item['唤醒音频']),
            'kws_text': item['唤醒文本'],
            'cmd_path': os.path.join(base_dir, item['识别音频']),
            'label': item.get('识别文本', None)
        }
        dataset.append(entry)
    return dataset


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    向音频信号添加高斯白噪声
    Args:
        signal: 原始音频信号
        snr_db: 信噪比（dB）
    Returns:
        添加噪声后的信号
    """
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-10:
        return signal
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal)).astype(np.float32)
    return signal + noise


def add_ambient_noise(signal: np.ndarray, snr_db: float,
                      noise_type: str = "white") -> np.ndarray:
    """
    添加不同类型的环境噪声
    Args:
        signal: 原始信号
        snr_db: 信噪比(dB)
        noise_type: 噪声类型 - "white", "pink", "babble"
    Returns:
        加噪后的信号
    """
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-10:
        return signal

    n = len(signal)
    if noise_type == "white":
        noise = np.random.randn(n).astype(np.float32)
    elif noise_type == "pink":
        # 粉红噪声：1/f频谱
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1  # 避免除零
        spectrum = 1.0 / np.sqrt(freqs)
        spectrum[0] = 0
        phases = np.random.uniform(0, 2 * np.pi, len(spectrum))
        noise = np.fft.irfft(spectrum * np.exp(1j * phases), n=n).astype(np.float32)
    elif noise_type == "babble":
        # 模拟人声嘈杂噪声（多频段随机）
        noise = np.zeros(n, dtype=np.float32)
        for _ in range(5):
            freq = np.random.uniform(100, 4000)
            t = np.arange(n) / 16000
            noise += np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)).astype(np.float32)
        noise += np.random.randn(n).astype(np.float32) * 0.3
    else:
        noise = np.random.randn(n).astype(np.float32)

    # 归一化噪声功率
    noise_power_actual = np.mean(noise ** 2)
    if noise_power_actual > 1e-10:
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power_actual)

    return signal + noise


def estimate_snr(signal: np.ndarray, sr: int = 16000,
                 frame_len: int = 512) -> float:
    """
    估计音频信号的信噪比
    使用能量最高帧与最低帧的比值来近似估算
    Args:
        signal: 音频信号
        sr: 采样率
        frame_len: 帧长
    Returns:
        估计的SNR（dB）
    """
    n_frames = len(signal) // frame_len
    if n_frames < 2:
        return 20.0  # 默认值

    frame_energies = []
    for i in range(n_frames):
        frame = signal[i * frame_len: (i + 1) * frame_len]
        energy = np.mean(frame ** 2)
        frame_energies.append(energy)

    frame_energies = sorted(frame_energies)
    # 取最低10%帧作为噪声估计，最高50%帧作为信号估计
    n_noise = max(1, n_frames // 10)
    n_signal = max(1, n_frames // 2)

    noise_energy = np.mean(frame_energies[:n_noise])
    signal_energy = np.mean(frame_energies[-n_signal:])

    if noise_energy < 1e-10:
        return 40.0  # 非常干净
    snr = 10 * np.log10(signal_energy / noise_energy)
    return float(snr)
