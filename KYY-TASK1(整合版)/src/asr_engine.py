#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR语音识别引擎模块
支持 FunASR (Paraformer) 和 OpenAI Whisper 两种引擎
Paraformer 针对中文优化，推理速度快
"""

import numpy as np
import os
from typing import Optional, List


class ASREngine:
    """ASR语音识别引擎"""

    def __init__(self, engine_type: str = "funasr",
                 model_name: str = "paraformer-zh",
                 device: str = "cpu",
                 cache_dir: str = None):
        """
        初始化ASR引擎
        Args:
            engine_type: 引擎类型 "funasr" 或 "whisper"
            model_name: 模型名称
            device: 推理设备
            cache_dir: 模型缓存目录
        """
        self.engine_type = engine_type
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载ASR模型"""
        if self.engine_type == "funasr":
            self._load_funasr()
        elif self.engine_type == "whisper":
            self._load_whisper()
        else:
            raise ValueError(f"不支持的ASR引擎: {self.engine_type}")

    def _load_funasr(self):
        """加载FunASR Paraformer模型"""
        try:
            from funasr import AutoModel
            print(f"[ASREngine] 正在加载 FunASR 模型: {self.model_name} ...")

            # 使用Paraformer中文模型
            self.model = AutoModel(
                model=self.model_name,
                device=self.device,
                disable_update=True,
            )
            print(f"[ASREngine] FunASR {self.model_name} 加载成功")
        except Exception as e:
            print(f"[ASREngine] FunASR加载失败: {e}")
            print("[ASREngine] 尝试回退到Whisper...")
            self.engine_type = "whisper"
            self._load_whisper()

    def _load_whisper(self):
        """加载OpenAI Whisper模型"""
        try:
            import whisper
            model_size = self.model_name if self.model_name != "paraformer-zh" else "base"
            print(f"[ASREngine] 正在加载 Whisper 模型: {model_size} ...")
            self.model = whisper.load_model(
                model_size,
                device=self.device,
                download_root=self.cache_dir
            )
            self.model_name = model_size
            print(f"[ASREngine] Whisper {model_size} 加载成功")
        except Exception as e:
            print(f"[ASREngine] Whisper加载失败: {e}")
            raise RuntimeError("无法加载任何ASR模型")

    def transcribe(self, audio: np.ndarray, sr: int = 16000,
                   language: str = "zh",
                   hotwords: str = "") -> str:
        """
        对单条音频进行语音识别
        Args:
            audio: 音频信号（float32 numpy数组）
            sr: 采样率
            language: 语言
            hotwords: 热词字符串（空格分隔）
        Returns:
            识别文本
        """
        if self.engine_type == "funasr":
            return self._transcribe_funasr(audio, sr, hotwords)
        elif self.engine_type == "whisper":
            return self._transcribe_whisper(audio, sr, language)
        return ""

    def _transcribe_funasr(self, audio: np.ndarray, sr: int = 16000,
                           hotwords: str = "") -> str:
        """FunASR识别"""
        try:
            kwargs = {
                "input": audio,
                "batch_size_s": 300,
            }
            if hotwords:
                kwargs["hotword"] = hotwords

            result = self.model.generate(**kwargs)

            if result and len(result) > 0:
                # FunASR返回格式: [{'key': ..., 'text': '...'}]
                text = result[0].get('text', '')
                return text.strip()
            return ""
        except Exception as e:
            print(f"[ASREngine] FunASR识别异常: {e}")
            return ""

    def _transcribe_whisper(self, audio: np.ndarray, sr: int = 16000,
                            language: str = "zh") -> str:
        """Whisper识别"""
        try:
            import whisper
            # Whisper需要float32, 16kHz
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # 填充/截断到30秒
            audio_padded = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio_padded).to(self.device)

            options = whisper.DecodingOptions(
                language=language,
                without_timestamps=True,
                fp16=False  # CPU不支持fp16
            )
            result = whisper.decode(self.model, mel, options)
            return result.text.strip()
        except Exception as e:
            print(f"[ASREngine] Whisper识别异常: {e}")
            return ""

    def transcribe_file(self, audio_path: str,
                        hotwords: str = "") -> str:
        """
        对音频文件进行识别
        Args:
            audio_path: 音频文件路径
            hotwords: 热词
        Returns:
            识别文本
        """
        if self.engine_type == "funasr":
            try:
                kwargs = {
                    "input": audio_path,
                    "batch_size_s": 300,
                }
                if hotwords:
                    kwargs["hotword"] = hotwords

                result = self.model.generate(**kwargs)
                if result and len(result) > 0:
                    text = result[0].get('text', '')
                    return text.strip()
                return ""
            except Exception as e:
                print(f"[ASREngine] 文件识别异常: {e}")
                return ""
        else:
            from src.data_loader import load_audio
            audio, sr = load_audio(audio_path)
            return self.transcribe(audio, sr)

    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {
            'engine_type': self.engine_type,
            'model_name': self.model_name,
            'device': self.device,
        }

        # 估算参数量
        if self.engine_type == "funasr":
            if "paraformer" in self.model_name.lower():
                info['params_approx'] = '220M'
            else:
                info['params_approx'] = 'unknown'
        elif self.engine_type == "whisper":
            whisper_params = {
                'tiny': '39M', 'base': '74M', 'small': '244M',
                'medium': '769M', 'large': '1550M', 'large-v3': '1550M'
            }
            info['params_approx'] = whisper_params.get(
                self.model_name, 'unknown')

        return info
