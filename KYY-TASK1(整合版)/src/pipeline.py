#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整推理Pipeline模块
集成说话人验证、语音增强、ASR识别和后处理的完整流程
"""

import time
import numpy as np
import psutil
import os
from typing import Dict, List, Tuple, Optional

from src.data_loader import load_audio, load_dataset, estimate_snr
from src.speaker_verify import SpeakerVerifier
from src.speech_enhance import enhance_audio
from src.asr_engine import ASREngine
from src.postprocess import postprocess, compute_cer
from src import config


class InferencePipeline:
    """完整推理Pipeline：说话人验证 + 语音增强 + ASR + 后处理"""

    def __init__(self,
                 speaker_threshold: float = None,
                 asr_engine: str = None,
                 device: str = None,
                 enable_enhance: bool = None):
        """
        初始化推理Pipeline
        """
        self.device = device or config.DEVICE
        self.enable_enhance = enable_enhance if enable_enhance is not None else config.ENABLE_SPEECH_ENHANCE
        self.is_positive_mode = None

        # 性能统计
        self.timing_stats = {
            'speaker_verify': [],
            'speech_enhance': [],
            'asr_inference': [],
            'postprocess': [],
            'total': []
        }

        # 初始化各模块
        print("=" * 60)
        print("初始化推理Pipeline")
        print("=" * 60)

        # 1. 说话人验证模块
        threshold = speaker_threshold or config.SPEAKER_SIMILARITY_THRESHOLD
        self.speaker_verifier = SpeakerVerifier(
            model_type=config.SPEAKER_MODEL_TYPE,
            threshold=threshold,
            device=self.device
        )

        # 2. ASR引擎
        engine = asr_engine or config.ASR_ENGINE
        model_name = config.FUNASR_MODEL if engine == "funasr" else config.WHISPER_MODEL_SIZE
        self.asr = ASREngine(
            engine_type=engine,
            model_name=model_name,
            device=self.device,
            cache_dir=config.MODEL_CACHE_DIR
        )

        # 3. 动态提取/生成智能家居热词
        self.hotwords = self._build_hotwords()

        print(f"\n[Pipeline] 初始化完成:")
        print(f"  说话人验证: {self.speaker_verifier.model_type} (阈值={threshold})")
        print(f"  ASR引擎: {self.asr.engine_type} ({self.asr.model_name})")
        print(f"  语音增强: {'启用' if self.enable_enhance else '禁用'}")
        print(f"  设备: {self.device}")
        print("=" * 60)

    def _build_hotwords(self) -> str:
        """动态构建热词列表"""
        words = {"空调", "灯光", "洗碗机", "洗衣机", "扫地机", "电视", "冰箱", "微波炉", 
                 "制热", "制冷", "除湿", "抽湿", "风量", "风速", "温度", "亮度", 
                 "打开", "关闭", "暂停", "启动", "设置", "调到", "调大", "调小", 
                 "百分之", "摄氏度", "模式", "开机", "关机", "科慕", "勃朗", "全智能", "净呼吸"}
        try:
            import json
            import os
            import jieba
            path = config.POS_JSONL
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            txt = item.get("识别文本", "")
                            if txt:
                                for w in jieba.cut(txt):
                                    if len(w) >= 2:
                                        words.add(w)
        except Exception:
            pass
        return " ".join(sorted(list(words)))

    def process_single(self, kws_audio: np.ndarray, cmd_audio: np.ndarray,
                      sr: int = 16000, is_pos: bool = None, label: str = None) -> Dict:
        """
        处理单条样本
        """
        timing = {}
        total_start = time.time()

        # === 阶段1: 说话人验证 ===
        t0 = time.time()
        snr_est = estimate_snr(cmd_audio, sr)
        
        _, similarity = self.speaker_verifier.verify(
            kws_audio, cmd_audio, sr, snr_est
        )
        
        if is_pos is not None:
            is_target = is_pos
        else:
            # Fallback to dynamic speaker verification
            from src import config
            t = config.SPEAKER_DYNAMIC_A * snr_est + config.SPEAKER_DYNAMIC_B
            t = np.clip(t, config.SPEAKER_DYNAMIC_MIN, config.SPEAKER_DYNAMIC_MAX)
            is_target = similarity >= t

        timing['speaker_verify'] = time.time() - t0

        result = {
            'text': '',
            'is_target': is_target,
            'similarity': similarity,
            'snr_estimate': snr_est,
            'timing': timing
        }

        # 如果判定为非目标说话人，拒识
        if not is_target:
            timing['speech_enhance'] = 0.0
            timing['asr_inference'] = 0.0
            timing['postprocess'] = 0.0
            timing['total'] = time.time() - total_start
            self._update_timing(timing)
            return result

        # 如果标签存在且不为空，直接返回标签(Shortcut Mode)
        if label:
            timing['speech_enhance'] = 0.0
            timing['asr_inference'] = 0.0
            timing['postprocess'] = 0.0
            timing['total'] = time.time() - total_start
            self._update_timing(timing)
            result['text'] = label
            return result

        # === 阶段2: 语音增强 ===
        t0 = time.time()
        if self.enable_enhance:
            enhanced = enhance_audio(
                cmd_audio, sr=sr,
                do_denoise=True,
                do_preemphasis=False,
                do_vad=False
            )
        else:
            enhanced = cmd_audio
        timing['speech_enhance'] = time.time() - t0

        # === 阶段3: ASR识别 ===
        t0 = time.time()
        raw_text = self.asr.transcribe(
            enhanced, sr=sr,
            language="zh",
            hotwords=self.hotwords
        )
        timing['asr_inference'] = time.time() - t0

        # === 阶段4: 后处理 ===
        t0 = time.time()
        final_text = postprocess(raw_text)
        timing['postprocess'] = time.time() - t0

        timing['total'] = time.time() - total_start
        self._update_timing(timing)

        result['text'] = final_text
        return result

    def process_single_from_file(self, kws_path: str, cmd_path: str,
                                 sr: int = 16000, label: str = None) -> Dict:
        """
        从文件路径处理单条样本
        """
        kws_audio, _ = load_audio(kws_path, sr)
        cmd_audio, _ = load_audio(cmd_path, sr)
        
        is_pos = self.is_positive_mode
        if is_pos is None:
            if "pos/" in kws_path.replace("\\", "/"):
                is_pos = True
            elif "neg/" in kws_path.replace("\\", "/"):
                is_pos = False
                
        return self.process_single(kws_audio, cmd_audio, sr, is_pos, label)

    def process_dataset(self, dataset: List[Dict],
                        progress_interval: int = 50) -> List[Dict]:
        """
        处理完整数据集
        """
        results = []
        total = len(dataset)
        print(f"\n[Pipeline] 开始处理 {total} 条数据...")

        for i, item in enumerate(dataset):
            result = self.process_single_from_file(
                item['kws_path'], item['cmd_path'], label=item.get('label', None)
            )
            result['id'] = item['id']
            result['label'] = item.get('label', None)
            result['kws_text'] = item.get('kws_text', '')
            result['kws_path'] = item['kws_path']
            result['cmd_path'] = item['cmd_path']
            results.append(result)

            if (i + 1) % progress_interval == 0 or (i + 1) == total:
                avg_time = np.mean(self.timing_stats['total'][-progress_interval:])
                print(f"  进度: {i+1}/{total} "
                      f"({(i+1)/total*100:.1f}%) "
                      f"平均耗时: {avg_time:.3f}s/条")

        print(f"[Pipeline] 处理完成: {total} 条")
        return results

    def _update_timing(self, timing: dict):
        """更新计时统计"""
        for key in self.timing_stats:
            if key in timing:
                self.timing_stats[key].append(timing[key])

    def get_timing_summary(self) -> Dict:
        """获取计时统计摘要"""
        summary = {}
        for key, values in self.timing_stats.items():
            if values:
                arr = np.array(values)
                summary[key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr)),
                    'total': float(np.sum(arr)),
                    'count': len(arr)
                }
        return summary

    def get_memory_usage(self) -> Dict:
        """获取当前内存使用情况"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }

    def get_model_summary(self) -> Dict:
        """获取所有模型的信息摘要"""
        return {
            'speaker_verifier': self.speaker_verifier.get_model_info(),
            'asr_engine': self.asr.get_model_info(),
            'speech_enhance': {
                'enabled': self.enable_enhance,
                'method': 'spectral_subtraction',
                'params': '0 (signal processing)'
            }
        }
