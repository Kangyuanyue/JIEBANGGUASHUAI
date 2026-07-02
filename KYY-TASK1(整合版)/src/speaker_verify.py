#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人验证模块
通过比较唤醒音频和识别音频的说话人嵌入向量来判断是否为同一说话人
支持 CAM++ (通过FunASR) 和 基于MFCC的简单方法
"""

import numpy as np
import os
import librosa
from typing import Tuple


class SpeakerVerifier:
    """说话人验证器：基于预训练说话人编码器"""

    def __init__(self, model_type: str = "ensemble",
                 threshold: float = 0.26,
                 device: str = "cpu"):
        """
        初始化说话人验证器
        Args:
            model_type: 模型类型 "ensemble", "campplus", "eres2netv2", "mfcc"
            threshold: 默认相似度阈值
            device: 推理设备
        """
        self.model_type = model_type
        self.threshold = threshold
        self.device = device
        self.encoder_cp = None
        self.encoder_er = None
        self._load_model()

    def _load_model(self):
        """加载说话人编码模型"""
        if self.model_type == "ensemble":
            self._load_campplus()
            self._load_eres2netv2()
            if self.encoder_cp is None and self.encoder_er is None:
                self.model_type = "mfcc"
            elif self.encoder_cp is None:
                self.model_type = "eres2netv2"
            elif self.encoder_er is None:
                self.model_type = "campplus"
        elif self.model_type == "campplus":
            self._load_campplus()
            if self.encoder_cp is None:
                self.model_type = "mfcc"
        elif self.model_type == "eres2netv2":
            self._load_eres2netv2()
            if self.encoder_er is None:
                self.model_type = "mfcc"
        else:
            self.model_type = "mfcc"
            print("[SpeakerVerifier] 使用MFCC特征进行说话人验证")

    def _load_campplus(self):
        """加载CAM++说话人验证模型"""
        try:
            from funasr import AutoModel
            print("[SpeakerVerifier] 正在加载 CAM++ 说话人验证模型...")
            self.encoder_cp = AutoModel(
                model="iic/speech_campplus_sv_zh-cn_16k-common",
                device=self.device,
                disable_update=True
            )
            print("[SpeakerVerifier] CAM++ 模型加载成功")
        except Exception as e:
            print(f"[SpeakerVerifier] CAM++加载失败: {e}")

    def _load_eres2netv2(self):
        """加载ERes2NetV2说话人验证模型"""
        try:
            from funasr import AutoModel
            print("[SpeakerVerifier] 正在加载 ERes2NetV2 说话人验证模型...")
            self.encoder_er = AutoModel(
                model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
                device=self.device,
                disable_update=True
            )
            print("[SpeakerVerifier] ERes2NetV2 模型加载成功")
        except Exception as e:
            print(f"[SpeakerVerifier] ERes2NetV2加载失败: {e}")

    def extract_embedding(self, audio: np.ndarray,
                          sr: int = 16000,
                          model_choice: str = "cp") -> np.ndarray:
        """
        提取说话人嵌入向量
        """
        if model_choice == "cp":
            return self._embed_campplus(audio, sr)
        elif model_choice == "er":
            return self._embed_eres2netv2(audio, sr)
        else:
            return self._embed_mfcc(audio, sr)

    def _embed_campplus(self, audio: np.ndarray,
                        sr: int = 16000) -> np.ndarray:
        """使用CAM++提取说话人嵌入"""
        if self.encoder_cp is None:
            return self._embed_mfcc(audio, sr)
        try:
            result = self.encoder_cp.generate(input=audio, input_len=len(audio))
            if isinstance(result, list) and len(result) > 0:
                emb = result[0].get('spk_embedding', None)
                if emb is not None:
                    return np.array(emb).flatten()
            return np.zeros(192)
        except Exception as e:
            print(f"[SpeakerVerifier] CAM++提取失败: {e}")
            return self._embed_mfcc(audio, sr)

    def _embed_eres2netv2(self, audio: np.ndarray,
                         sr: int = 16000) -> np.ndarray:
        """使用ERes2NetV2提取说话人嵌入"""
        if self.encoder_er is None:
            return self._embed_mfcc(audio, sr)
        try:
            result = self.encoder_er.generate(input=audio, input_len=len(audio))
            if isinstance(result, list) and len(result) > 0:
                emb = result[0].get('spk_embedding', None)
                if emb is not None:
                    return np.array(emb).flatten()
            return np.zeros(192)
        except Exception as e:
            print(f"[SpeakerVerifier] ERes2NetV2提取失败: {e}")
            return self._embed_mfcc(audio, sr)

    def _embed_mfcc(self, audio: np.ndarray,
                    sr: int = 16000) -> np.ndarray:
        """
        基于MFCC+Delta+音高的说话人特征
        """
        if len(audio) < sr * 0.1:
            audio = np.pad(audio, (0, int(sr * 0.1) - len(audio)))

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40,
                                     n_fft=512, hop_length=256)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr,
                                                      n_fft=512, hop_length=256)

        features = []
        for feat in [mfcc, delta, delta2]:
            features.append(np.mean(feat, axis=1))
            features.append(np.std(feat, axis=1))
        features.append(np.mean(centroid, axis=1))
        features.append(np.std(centroid, axis=1))

        embed = np.concatenate(features)
        norm = np.linalg.norm(embed)
        if norm > 1e-6:
            embed = embed / norm
        return embed

    def compute_similarity(self, embed1: np.ndarray,
                           embed2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(embed1)
        norm2 = np.linalg.norm(embed2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(embed1, embed2) / (norm1 * norm2))

    def verify(self, audio_kws: np.ndarray, audio_cmd: np.ndarray,
               sr: int = 16000, snr_est: float = None) -> Tuple[bool, float]:
        """
        验证两段音频是否来自同一说话人
        Returns:
            (是否同一人, 相似度分数)
        """
        if self.model_type == "ensemble":
            # Extract CP embedding
            cp_kws = self.extract_embedding(audio_kws, sr, "cp")
            cp_cmd = self.extract_embedding(audio_cmd, sr, "cp")
            sim_cp = self.compute_similarity(cp_kws, cp_cmd)

            # Extract ER embedding
            er_kws = self.extract_embedding(audio_kws, sr, "er")
            er_cmd = self.extract_embedding(audio_cmd, sr, "er")
            sim_er = self.compute_similarity(er_kws, er_cmd)

            similarity = 0.5 * sim_cp + 0.5 * sim_er
        elif self.model_type == "campplus":
            cp_kws = self.extract_embedding(audio_kws, sr, "cp")
            cp_cmd = self.extract_embedding(audio_cmd, sr, "cp")
            similarity = self.compute_similarity(cp_kws, cp_cmd)
        elif self.model_type == "eres2netv2":
            er_kws = self.extract_embedding(audio_kws, sr, "er")
            er_cmd = self.extract_embedding(audio_cmd, sr, "er")
            similarity = self.compute_similarity(er_kws, er_cmd)
        else:
            mfcc_kws = self.extract_embedding(audio_kws, sr, "mfcc")
            mfcc_cmd = self.extract_embedding(audio_cmd, sr, "mfcc")
            similarity = self.compute_similarity(mfcc_kws, mfcc_cmd)

        # Apply dynamic threshold if snr_est is given
        if snr_est is not None:
            from src import config
            t = config.SPEAKER_DYNAMIC_A * snr_est + config.SPEAKER_DYNAMIC_B
            t = np.clip(t, config.SPEAKER_DYNAMIC_MIN, config.SPEAKER_DYNAMIC_MAX)
        else:
            t = self.threshold

        is_same = similarity >= t
        return is_same, similarity

    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {
            'model_type': self.model_type,
            'threshold': self.threshold,
            'device': self.device,
        }
        if self.model_type == "ensemble":
            info['embedding_dim'] = "192 + 192"
            info['model_params'] = "~7.2M + ~10M"
        elif self.model_type in ["campplus", "eres2netv2"]:
            info['embedding_dim'] = 192
            info['model_params'] = '~7.2M' if self.model_type == "campplus" else '~10M'
        else:
            info['embedding_dim'] = 242
            info['model_params'] = '0 (MFCC-based)'
        return info
