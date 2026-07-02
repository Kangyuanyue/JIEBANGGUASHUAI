#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音增强/降噪模块
提供频谱减法降噪、预加重和语音活动检测（VAD）
在ASR之前对音频进行预处理，提升识别鲁棒性
"""

import numpy as np
from typing import Tuple, Optional


def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    预加重滤波器：增强高频成分
    Args:
        signal: 输入音频信号
        coeff: 预加重系数（0.95~0.97）
    Returns:
        预加重后的信号
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def spectral_subtraction(signal: np.ndarray, sr: int = 16000,
                         frame_len: int = 512, hop_len: int = 256,
                         alpha: float = 2.0, beta: float = 0.01,
                         noise_frames: int = 5) -> np.ndarray:
    """
    频谱减法降噪
    使用前几帧估计噪声频谱，从整体频谱中减去
    Args:
        signal: 输入音频信号
        sr: 采样率
        frame_len: FFT帧长
        hop_len: 帧移
        alpha: 过减因子（越大降噪越强，但可能引入失真）
        beta: 频谱下限（防止频谱为负值）
        noise_frames: 用于噪声估计的前N帧
    Returns:
        降噪后的信号
    """
    if len(signal) < frame_len:
        return signal

    # 分帧
    n_frames = (len(signal) - frame_len) // hop_len + 1
    if n_frames < noise_frames + 1:
        return signal

    # 汉宁窗
    window = np.hanning(frame_len)

    # STFT
    frames = []
    for i in range(n_frames):
        start = i * hop_len
        frame = signal[start:start + frame_len] * window
        frames.append(frame)
    frames = np.array(frames)

    # FFT
    spectra = np.fft.rfft(frames, axis=1)
    magnitudes = np.abs(spectra)
    phases = np.angle(spectra)

    # 估计噪声频谱（使用前N帧的平均幅度谱）
    noise_spectrum = np.mean(magnitudes[:noise_frames], axis=0)

    # 频谱减法
    clean_magnitudes = magnitudes.copy()
    for i in range(n_frames):
        subtracted = magnitudes[i] ** 2 - alpha * noise_spectrum ** 2
        # 频谱下限：防止过度减法导致的负值
        floor = beta * magnitudes[i] ** 2
        clean_magnitudes[i] = np.sqrt(np.maximum(subtracted, floor))

    # 重建信号（使用原始相位）
    clean_spectra = clean_magnitudes * np.exp(1j * phases)
    clean_frames = np.fft.irfft(clean_spectra, n=frame_len, axis=1)

    # 重叠相加(OLA)合成
    output_len = (n_frames - 1) * hop_len + frame_len
    output = np.zeros(output_len, dtype=np.float32)
    window_sum = np.zeros(output_len, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_len
        output[start:start + frame_len] += clean_frames[i] * window
        window_sum[start:start + frame_len] += window ** 2

    # 归一化
    window_sum = np.maximum(window_sum, 1e-8)
    output = output / window_sum

    # 裁剪到原始长度
    output = output[:len(signal)]

    return output.astype(np.float32)


def simple_vad(signal: np.ndarray, sr: int = 16000,
               frame_len_ms: int = 25, threshold_db: float = -40
               ) -> np.ndarray:
    """
    简单的基于能量的语音活动检测
    Args:
        signal: 输入音频信号
        sr: 采样率
        frame_len_ms: 帧长（毫秒）
        threshold_db: 能量阈值（dB）
    Returns:
        去除静音后的音频信号
    """
    frame_len = int(sr * frame_len_ms / 1000)
    n_frames = len(signal) // frame_len

    if n_frames < 1:
        return signal

    # 计算每帧能量
    energies = []
    for i in range(n_frames):
        frame = signal[i * frame_len: (i + 1) * frame_len]
        energy = np.mean(frame ** 2)
        energies.append(energy)

    energies = np.array(energies)
    max_energy = np.max(energies)
    if max_energy < 1e-10:
        return signal

    # 转换为dB
    energies_db = 10 * np.log10(energies / max_energy + 1e-10)

    # 标记语音帧
    voice_frames = energies_db > threshold_db

    # 拼接语音帧
    voice_signal = []
    for i in range(n_frames):
        if voice_frames[i]:
            start = i * frame_len
            voice_signal.append(signal[start:start + frame_len])

    if len(voice_signal) == 0:
        return signal  # 全静音时返回原信号

    return np.concatenate(voice_signal)


def enhance_audio(signal: np.ndarray, sr: int = 16000,
                  do_denoise: bool = True,
                  do_preemphasis: bool = False,
                  do_vad: bool = False) -> np.ndarray:
    """
    完整的语音增强处理流程
    Args:
        signal: 原始音频信号
        sr: 采样率
        do_denoise: 是否进行降噪
        do_preemphasis: 是否进行预加重
        do_vad: 是否进行VAD
    Returns:
        增强后的音频信号
    """
    output = signal.copy()

    # 1. 频谱减法降噪
    if do_denoise:
        output = spectral_subtraction(
            output, sr=sr,
            alpha=2.0, beta=0.01,
            noise_frames=5
        )

    # 2. 预加重
    if do_preemphasis:
        output = pre_emphasis(output, coeff=0.97)

    # 3. VAD去静音
    if do_vad:
        output = simple_vad(output, sr=sr)

    # 4. 幅度归一化
    max_val = np.max(np.abs(output))
    if max_val > 1e-6:
        output = output / max_val * 0.95

    return output
