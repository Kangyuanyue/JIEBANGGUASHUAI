import argparse
import csv
import gc
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from funasr import AutoModel


# ============================================================
#   FUN-ASR/datasetA/pos.jsonl
#   FUN-ASR/datasetA/neg.jsonl
# ============================================================
ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_ROOT = ROOT / "datasetA"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "funasr_sv_gate_datasetA"

ASR_MODEL_NAME = "FunAudioLLM/Fun-ASR-Nano-2512"
SV_MODEL_NAME = "iic/speech_eres2netv2_sv_zh-cn_16k-common"

DEFAULT_ASR_DEVICE = "cuda:0"
DEFAULT_SV_DEVICE = "cuda:0"

# score >= threshold：接受，进入 ASR
# score < threshold ：拒识，最终预测文本置空
DEFAULT_THRESHOLD = 0.30


# ============================================================
# 通用工具
# ============================================================
def repair_key(key: Any) -> Any:
    """尝试修复中文字段名乱码，正常字段名原样返回。"""
    if not isinstance(key, str):
        return key

    try:
        return key.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return key


def get_field(item: Dict[str, Any], *field_names: str) -> Any:
    """同时兼容正常字段名和乱码字段名。"""
    for field_name in field_names:
        for key, value in item.items():
            if key == field_name or repair_key(key) == field_name:
                return value
    return None


def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """支持 utf-8-sig、utf-8 和 gb18030。"""
    encodings = ["utf-8-sig", "utf-8", "gb18030"]
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            records: List[Dict[str, Any]] = []

            with open(jsonl_path, "r", encoding=encoding) as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{jsonl_path.name} 第 {line_no} 行 JSON 格式错误：{error}"
                        ) from error

                    if not isinstance(item, dict):
                        raise ValueError(
                            f"{jsonl_path.name} 第 {line_no} 行不是 JSON 对象"
                        )

                    records.append(item)

            print(
                f"读取成功：{jsonl_path.name}，"
                f"编码={encoding}，样本数={len(records)}"
            )
            return records

        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(f"无法读取 {jsonl_path}，最后错误：{last_error}")


def resolve_audio_path(data_root: Path, path_value: Any) -> Optional[Path]:
    """将 JSONL 中的相对路径转换成绝对路径。"""
    if path_value is None:
        return None

    path_text = str(path_value).strip()
    if not path_text:
        return None


    path_text = path_text.replace("\\", "/")
    path = Path(path_text)

    if not path.is_absolute():
        path = data_root / path

    return path.resolve()


def release_cuda_memory() -> None:
    """释放模型和 CUDA 缓存。"""
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# 兼容两种 DatasetA JSONL 格式
# ============================================================
def parse_dataset_item(
    item: Dict[str, Any],
    split_name: str,
    index: int,
    data_root: Path,
) -> Dict[str, Any]:
    """
    兼容以下两种格式。

    格式一：
    {
        "input": {
            "kws_path": "pos/kws_0.wav",
            "kws_txt": "你好科慕",
            "cmd_path": "pos/cmd_0.wav"
        },
        "label": "空调开到制热..."
    }

    格式二：
    {
        "唤醒音频": "pos/kws_0.wav",
        "唤醒文本": "你好科慕",
        "识别音频": "pos/cmd_0.wav",
        "识别文本": "空调开到制热..."
    }
    """
    input_info = get_field(item, "input")

    if isinstance(input_info, dict):
        wake_rel_path = get_field(
            input_info,
            "kws_path",
            "wake_path",
            "唤醒音频",
        )
        wake_text = get_field(
            input_info,
            "kws_txt",
            "wake_text",
            "唤醒文本",
        )
        cmd_rel_path = get_field(
            input_info,
            "cmd_path",
            "command_path",
            "识别音频",
        )
        ref_text = get_field(
            item,
            "label",
            "识别文本",
            "target",
            "text",
        )
    else:
        wake_rel_path = get_field(
            item,
            "唤醒音频",
            "kws_path",
            "wake_path",
        )
        wake_text = get_field(
            item,
            "唤醒文本",
            "kws_txt",
            "wake_text",
        )
        cmd_rel_path = get_field(
            item,
            "识别音频",
            "cmd_path",
            "command_path",
        )
        ref_text = get_field(
            item,
            "识别文本",
            "label",
            "target",
            "text",
        )

    wake_path = resolve_audio_path(data_root, wake_rel_path)
    cmd_path = resolve_audio_path(data_root, cmd_rel_path)

    missing_reasons: List[str] = []

    if wake_path is None:
        missing_reasons.append("缺少唤醒音频字段")
    elif not wake_path.exists():
        missing_reasons.append(f"唤醒音频不存在：{wake_path}")

    if cmd_path is None:
        missing_reasons.append("缺少识别音频字段")
    elif not cmd_path.exists():
        missing_reasons.append(f"识别音频不存在：{cmd_path}")

    if split_name == "pos" and ref_text is None:
        missing_reasons.append("Pos 样本缺少参考文本")

    return {
        "split": split_name,
        "index": index,
        "id": item.get("id"),
        "wake_rel_path": None if wake_rel_path is None else str(wake_rel_path),
        "wake_text": wake_text,
        "cmd_rel_path": None if cmd_rel_path is None else str(cmd_rel_path),
        "reference_text": ref_text,
        "wake_path": None if wake_path is None else str(wake_path),
        "cmd_path": None if cmd_path is None else str(cmd_path),
        "valid_audio": len(missing_reasons) == 0,
        "missing_reason": "；".join(missing_reasons) if missing_reasons else None,
        # 后续推理字段
        "speaker_score": None,
        "sv_success": False,
        "sv_error": None,
        "asr_text": "",
        "asr_success": False,
        "asr_error": None,
    }


def prepare_records(
    data_root: Path,
    pos_records: List[Dict[str, Any]],
    neg_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """解析 Pos 和 Neg 数据。"""
    parsed: List[Dict[str, Any]] = []

    for index, item in enumerate(pos_records, start=1):
        parsed.append(
            parse_dataset_item(
                item=item,
                split_name="pos",
                index=index,
                data_root=data_root,
            )
        )

    for index, item in enumerate(neg_records, start=1):
        parsed.append(
            parse_dataset_item(
                item=item,
                split_name="neg",
                index=index,
                data_root=data_root,
            )
        )

    valid_count = sum(1 for item in parsed if item["valid_audio"])
    missing_count = len(parsed) - valid_count

    print(
        f"数据解析完成：总样本={len(parsed)}，"
        f"有效样本={valid_count}，缺失/异常样本={missing_count}"
    )

    if missing_count:
        for item in parsed:
            if not item["valid_audio"]:
                print(
                    f"[跳过] {item['split']} 第 {item['index']} 条："
                    f"{item['missing_reason']}"
                )

    return parsed


# ============================================================
# 官方 CER 文本归一化
# ============================================================
def normalize_text(text: Any) -> str:
    """
    比赛官方规则：
    1. Unicode NFKC；
    2. 转小写；
    3. 去前后空白；
    4. 删除所有空白字符；
    5. 删除所有 Unicode P* 标点字符。
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()

    normalized_chars: List[str] = []

    for char in text:
        if char.isspace():
            continue

        if unicodedata.category(char).startswith("P"):
            continue

        normalized_chars.append(char)

    return "".join(normalized_chars)


# ============================================================
# Levenshtein：统计 S、I、D
# ============================================================
def levenshtein_details(
    reference: str,
    hypothesis: str,
) -> Dict[str, int]:
    """
    返回字符级：
    S：替换
    I：插入
    D：删除
    errors = S + I + D
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    dp = [
        [0 for _ in range(hyp_len + 1)]
        for _ in range(ref_len + 1)
    ]
    operation = [
        ["" for _ in range(hyp_len + 1)]
        for _ in range(ref_len + 1)
    ]

    for i in range(1, ref_len + 1):
        dp[i][0] = i
        operation[i][0] = "D"

    for j in range(1, hyp_len + 1):
        dp[0][j] = j
        operation[0][j] = "I"

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                operation[i][j] = "E"
                continue

            substitution_cost = dp[i - 1][j - 1] + 1
            deletion_cost = dp[i - 1][j] + 1
            insertion_cost = dp[i][j - 1] + 1

            best_cost = min(
                substitution_cost,
                deletion_cost,
                insertion_cost,
            )
            dp[i][j] = best_cost

            # 多种最优路径同时存在时，只影响 S/I/D 拆分，
            # 不影响总编辑距离和最终 CER。
            if best_cost == substitution_cost:
                operation[i][j] = "S"
            elif best_cost == deletion_cost:
                operation[i][j] = "D"
            else:
                operation[i][j] = "I"

    substitutions = 0
    insertions = 0
    deletions = 0

    i = ref_len
    j = hyp_len

    while i > 0 or j > 0:
        op = operation[i][j]

        if op == "E":
            i -= 1
            j -= 1
        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            i -= 1
        elif op == "I":
            insertions += 1
            j -= 1
        else:
            raise RuntimeError(
                f"Levenshtein 回溯失败：i={i}, j={j}, op={op!r}"
            )

    errors = substitutions + insertions + deletions

    return {
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "errors": errors,
    }


class CERMetric:
    """整体 CER = 总(S+I+D) / 总参考字符数。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_chars = 0
        self.total_errors = 0
        self.total_substitutions = 0
        self.total_insertions = 0
        self.total_deletions = 0

    def update(self, pred: Any, target: Any) -> Dict[str, Any]:
        original_pred = "" if pred is None else str(pred)
        original_target = "" if target is None else str(target)

        norm_pred = normalize_text(original_pred)
        norm_target = normalize_text(original_target)

        detail = levenshtein_details(
            reference=norm_target,
            hypothesis=norm_pred,
        )

        target_chars = len(norm_target)
        errors = detail["errors"]

        if target_chars == 0:
            sample_cer = 0.0 if errors == 0 else 1.0
        else:
            sample_cer = errors / target_chars

        self.total_chars += target_chars
        self.total_errors += errors
        self.total_substitutions += detail["substitutions"]
        self.total_insertions += detail["insertions"]
        self.total_deletions += detail["deletions"]

        return {
            "orig_pred": original_pred,
            "orig_target": original_target,
            "norm_pred": norm_pred,
            "norm_target": norm_target,
            "target_chars": target_chars,
            "substitutions": detail["substitutions"],
            "insertions": detail["insertions"],
            "deletions": detail["deletions"],
            "errors": errors,
            "cer": sample_cer,
        }

    def compute(self) -> Dict[str, Any]:
        if self.total_chars == 0:
            cer = 0.0 if self.total_errors == 0 else 1.0
        else:
            cer = self.total_errors / self.total_chars

        return {
            "CER": cer,
            "total_ref_chars": self.total_chars,
            "total_char_errors": self.total_errors,
            "total_substitutions": self.total_substitutions,
            "total_insertions": self.total_insertions,
            "total_deletions": self.total_deletions,
        }


# ============================================================
# 声纹模型：唤醒音频 vs 命令混合音频
# ============================================================
def embedding_to_numpy(embedding: Any) -> np.ndarray:
    """兼容 torch.Tensor、numpy.ndarray 和普通列表。"""
    if embedding is None:
        raise ValueError("声纹模型没有返回 spk_embedding")

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if vector.size == 0:
        raise ValueError("声纹向量为空")

    if not np.all(np.isfinite(vector)):
        raise ValueError("声纹向量包含 NaN 或 Inf")

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("声纹向量范数为 0")

    return vector / norm


def extract_speaker_embedding(
    sv_model: AutoModel,
    wav_path: Path,
) -> np.ndarray:
    """使用 ERes2NetV2 提取归一化声纹向量。"""
    result = sv_model.generate(input=str(wav_path))

    if not result:
        raise RuntimeError("声纹模型返回空结果")

    first_result = result[0]

    if not isinstance(first_result, dict):
        raise TypeError(
            f"声纹结果格式错误：{type(first_result).__name__}"
        )

    embedding = first_result.get("spk_embedding")
    return embedding_to_numpy(embedding)


def cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """输入已经 L2 归一化，点积即余弦相似度。"""
    score = float(np.dot(embedding_a, embedding_b))
    return float(np.clip(score, -1.0, 1.0))


def run_speaker_scoring(
    records: List[Dict[str, Any]],
    model_name: str,
    device: str,
    hub: str,
) -> None:
    """计算每条样本的唤醒词—命令音频声纹相似度。"""
    print("\n========== 加载短音频声纹模型 ==========")
    print(f"声纹模型：{model_name}")
    print(f"声纹设备：{device}")

    sv_model = AutoModel(
        model=model_name,
        device=device,
        hub=hub,
    )

    valid_records = [item for item in records if item["valid_audio"]]
    total = len(valid_records)

    for count, item in enumerate(valid_records, start=1):
        wake_path = Path(item["wake_path"])
        cmd_path = Path(item["cmd_path"])

        try:
            wake_embedding = extract_speaker_embedding(
                sv_model=sv_model,
                wav_path=wake_path,
            )
            cmd_embedding = extract_speaker_embedding(
                sv_model=sv_model,
                wav_path=cmd_path,
            )

            item["speaker_score"] = cosine_similarity(
                wake_embedding,
                cmd_embedding,
            )
            item["sv_success"] = True
            item["sv_error"] = None

        except Exception as error:
            item["speaker_score"] = None
            item["sv_success"] = False
            item["sv_error"] = (
                f"{type(error).__name__}: {error}"
            )

            print(
                f"[声纹失败] {item['split']} 第 {item['index']} 条："
                f"{item['sv_error']}"
            )

        if count % 20 == 0 or count == total:
            print(f"声纹评分：已完成 {count}/{total}")

    del sv_model
    release_cuda_memory()


# ============================================================
# ASR 推理
# ============================================================
def recognize_one(
    asr_model: AutoModel,
    wav_path: Path,
) -> Tuple[bool, str, Optional[str]]:
    """返回：推理是否成功、识别文本、错误信息。"""
    try:
        result = asr_model.generate(
            input=[str(wav_path)],
            cache={},
            batch_size=1,
            language="中文",
            itn=False,
        )

        if not result:
            return True, "", None

        first_result = result[0]

        if isinstance(first_result, dict):
            text = first_result.get("text", "")
        else:
            text = str(first_result)

        return True, "" if text is None else str(text).strip(), None

    except Exception as error:
        return (
            False,
            "",
            f"{type(error).__name__}: {error}",
        )


def run_asr(
    records: List[Dict[str, Any]],
    model_name: str,
    device: str,
    hub: str,
    remote_code_path: Optional[Path],
) -> None:
    """
    对所有有效命令音频执行一次 ASR。

    之后切换阈值时直接复用原始识别结果，
    不必反复运行大模型。
    """
    print("\n========== 加载 Fun-ASR 模型 ==========")
    print(f"ASR 模型：{model_name}")
    print(f"ASR 设备：{device}")

    model_kwargs: Dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "device": device,
        "hub": hub,
    }

    if remote_code_path is not None and remote_code_path.exists():
        model_kwargs["remote_code"] = str(remote_code_path)
        print(f"使用本地 remote_code：{remote_code_path}")
    else:
        print("未找到本地 model.py，不传 remote_code 文件路径")

    asr_model = AutoModel(**model_kwargs)

    valid_records = [item for item in records if item["valid_audio"]]
    total = len(valid_records)

    for count, item in enumerate(valid_records, start=1):
        cmd_path = Path(item["cmd_path"])

        success, text, error = recognize_one(
            asr_model=asr_model,
            wav_path=cmd_path,
        )

        item["asr_success"] = success
        item["asr_text"] = text
        item["asr_error"] = error

        if not success:
            print(
                f"[ASR 失败] {item['split']} 第 {item['index']} 条："
                f"{error}"
            )

        if count % 20 == 0 or count == total:
            print(f"ASR：已完成 {count}/{total}")

    del asr_model
    release_cuda_memory()


# ============================================================
# 原始模型输出缓存
# ============================================================
def save_raw_cache(
    records: List[Dict[str, Any]],
    cache_file: Path,
) -> None:
    """保存声纹分数和未经过门控的 ASR 输出。"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"原始推理缓存已保存：{cache_file}")


def load_raw_cache(cache_file: Path) -> List[Dict[str, Any]]:
    """读取原始模型输出缓存。"""
    records: List[Dict[str, Any]] = []

    with open(cache_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"缓存文件第 {line_no} 行损坏：{error}"
                ) from error

    print(
        f"已复用原始推理缓存：{cache_file}，"
        f"样本数={len(records)}"
    )
    return records


# ============================================================
# 阈值门控与指标
# ============================================================
def apply_threshold(
    item: Dict[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """
    门控规则：

    1. 声纹失败：
       系统失败，最终文本置空。

    2. score < threshold：
       系统主动拒识，最终文本置空；
       此时不依赖 ASR 是否成功。

    3. score >= threshold：
       接受并采用 ASR 输出；
       若 ASR 推理失败，则系统失败，最终文本置空。
    """
    score = item.get("speaker_score")
    sv_success = bool(item.get("sv_success", False))

    if not sv_success or score is None:
        return {
            "accepted": False,
            "gate_rejected": False,
            "pipeline_success": False,
            "final_text": "",
            "decision_reason": "speaker_verification_failed",
        }

    accepted = float(score) >= threshold

    if not accepted:
        return {
            "accepted": False,
            "gate_rejected": True,
            "pipeline_success": True,
            "final_text": "",
            "decision_reason": "score_below_threshold",
        }

    asr_success = bool(item.get("asr_success", False))

    if not asr_success:
        return {
            "accepted": True,
            "gate_rejected": False,
            "pipeline_success": False,
            "final_text": "",
            "decision_reason": "asr_failed_after_accept",
        }

    return {
        "accepted": True,
        "gate_rejected": False,
        "pipeline_success": True,
        "final_text": str(item.get("asr_text", "")),
        "decision_reason": "accepted",
    }


def threshold_tag(threshold: float) -> str:
    """将 0.30 转成 0p300，便于文件命名。"""
    return f"{threshold:.3f}".replace("-", "m").replace(".", "p")


def evaluate_threshold(
    records: List[Dict[str, Any]],
    threshold: float,
    output_dir: Optional[Path] = None,
    save_predictions: bool = False,
) -> Dict[str, Any]:
    """在固定阈值下计算 Pos CER 和 Neg RR。"""
    metric = CERMetric()

    pos_valid_count = 0
    neg_valid_count = 0

    pos_accept_count = 0
    pos_gate_reject_count = 0
    neg_accept_count = 0
    neg_gate_reject_count = 0

    pos_pipeline_failed_count = 0
    neg_pipeline_failed_count = 0

    neg_correct_reject_count = 0

    pos_output_rows: List[Dict[str, Any]] = []
    neg_output_rows: List[Dict[str, Any]] = []

    for item in records:
        if not item.get("valid_audio", False):
            continue

        decision = apply_threshold(item, threshold)
        final_text = decision["final_text"]

        common_output = {
            "index": item["index"],
            "id": item.get("id"),
            "唤醒音频": item.get("wake_rel_path"),
            "唤醒文本": item.get("wake_text"),
            "识别音频": item.get("cmd_rel_path"),
            "参考文本": item.get("reference_text"),
            "未门控ASR文本": item.get("asr_text", ""),
            "最终预测文本": final_text,
            "声纹相似度": item.get("speaker_score"),
            "门控阈值": threshold,
            "是否通过阈值": decision["accepted"],
            "是否由门控拒识": decision["gate_rejected"],
            "系统是否正常完成": decision["pipeline_success"],
            "决策原因": decision["decision_reason"],
            "声纹推理是否成功": item.get("sv_success", False),
            "声纹错误": item.get("sv_error"),
            "ASR推理是否成功": item.get("asr_success", False),
            "ASR错误": item.get("asr_error"),
        }

        if item["split"] == "pos":
            pos_valid_count += 1

            if decision["accepted"]:
                pos_accept_count += 1

            if decision["gate_rejected"]:
                pos_gate_reject_count += 1

            if not decision["pipeline_success"]:
                pos_pipeline_failed_count += 1

            sample = metric.update(
                pred=final_text,
                target=item.get("reference_text", ""),
            )

            common_output.update({
                "规范化参考文本": sample["norm_target"],
                "规范化预测文本": sample["norm_pred"],
                "参考字符数N": sample["target_chars"],
                "替换错误S": sample["substitutions"],
                "插入错误I": sample["insertions"],
                "删除错误D": sample["deletions"],
                "字符错误总数": sample["errors"],
                "单句CER": round(sample["cer"], 8),
            })
            pos_output_rows.append(common_output)

        else:
            neg_valid_count += 1

            if decision["accepted"]:
                neg_accept_count += 1

            if decision["gate_rejected"]:
                neg_gate_reject_count += 1

            if not decision["pipeline_success"]:
                neg_pipeline_failed_count += 1

            # 最终系统输出为空，才表示负样本被拒识；
            # 但推理异常不能冒充正确拒识。
            is_correct_reject = (
                decision["pipeline_success"]
                and normalize_text(final_text) == ""
            )

            if is_correct_reject:
                neg_correct_reject_count += 1

            common_output.update({
                "规范化最终预测文本": normalize_text(final_text),
                "是否正确拒识": is_correct_reject,
            })
            neg_output_rows.append(common_output)

    cer_result = metric.compute()

    rr = (
        neg_correct_reject_count / neg_valid_count
        if neg_valid_count > 0
        else 0.0
    )

    pos_accept_rate = (
        pos_accept_count / pos_valid_count
        if pos_valid_count > 0
        else 0.0
    )

    neg_accept_rate = (
        neg_accept_count / neg_valid_count
        if neg_valid_count > 0
        else 0.0
    )

    neg_gate_reject_rate = (
        neg_gate_reject_count / neg_valid_count
        if neg_valid_count > 0
        else 0.0
    )

    result = {
        "threshold": threshold,
        "pos": {
            "valid_count": pos_valid_count,
            "accept_count": pos_accept_count,
            "accept_rate": pos_accept_rate,
            "gate_reject_count": pos_gate_reject_count,
            "pipeline_failed_count": pos_pipeline_failed_count,
            **cer_result,
        },
        "neg": {
            "valid_count": neg_valid_count,
            "accept_count": neg_accept_count,
            "accept_rate": neg_accept_rate,
            "gate_reject_count": neg_gate_reject_count,
            "gate_reject_rate": neg_gate_reject_rate,
            "correct_reject_count": neg_correct_reject_count,
            "pipeline_failed_count": neg_pipeline_failed_count,
            "RR": rr,
        },
    }

    if save_predictions:
        if output_dir is None:
            raise ValueError("save_predictions=True 时必须提供 output_dir")

        output_dir.mkdir(parents=True, exist_ok=True)
        tag = threshold_tag(threshold)

        pos_file = output_dir / f"pos_prediction_threshold_{tag}.jsonl"
        neg_file = output_dir / f"neg_prediction_threshold_{tag}.jsonl"
        summary_file = output_dir / f"summary_threshold_{tag}.json"

        with open(pos_file, "w", encoding="utf-8") as f:
            for row in pos_output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(neg_file, "w", encoding="utf-8") as f:
            for row in neg_output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        result["pos"]["prediction_file"] = str(pos_file)
        result["neg"]["prediction_file"] = str(neg_file)

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["summary_file"] = str(summary_file)

    return result


def print_evaluation(result: Dict[str, Any], title: str) -> None:
    """打印一个阈值下的指标。"""
    pos = result["pos"]
    neg = result["neg"]

    print(f"\n========== {title} ==========")
    print(f"声纹阈值：{result['threshold']:.6f}")

    print("\n----- Pos CER -----")
    print(f"有效样本数：{pos['valid_count']}")
    print(
        f"通过阈值：{pos['accept_count']}/"
        f"{pos['valid_count']} "
        f"({pos['accept_rate']:.4%})"
    )
    print(f"替换错误 S：{pos['total_substitutions']}")
    print(f"插入错误 I：{pos['total_insertions']}")
    print(f"删除错误 D：{pos['total_deletions']}")
    print(f"总错误数：{pos['total_char_errors']}")
    print(f"总参考字符数 N：{pos['total_ref_chars']}")
    print(f"Pos CER：{pos['CER']:.6f} ({pos['CER']:.4%})")

    print("\n----- Neg RR -----")
    print(f"有效样本数：{neg['valid_count']}")
    print(
        f"阈值直接拒识：{neg['gate_reject_count']}/"
        f"{neg['valid_count']} "
        f"({neg['gate_reject_rate']:.4%})"
    )
    print(
        f"最终正确拒识：{neg['correct_reject_count']}/"
        f"{neg['valid_count']}"
    )
    print(f"Neg RR：{neg['RR']:.6f} ({neg['RR']:.4%})")


# ============================================================
# 自动搜索最佳阈值
# ============================================================
def make_thresholds(
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> List[float]:
    if threshold_step <= 0:
        raise ValueError("threshold_step 必须大于 0")

    if threshold_max < threshold_min:
        raise ValueError("threshold_max 不能小于 threshold_min")

    count = int(
        round((threshold_max - threshold_min) / threshold_step)
    ) + 1

    return [
        round(threshold_min + i * threshold_step, 10)
        for i in range(count)
    ]


def search_best_threshold(
    records: List[Dict[str, Any]],
    output_dir: Path,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    max_cer_increase: float,
) -> Dict[str, Any]:
    """
    搜索策略：

    1. baseline：阈值低于余弦相似度最小值，等价于不做声纹门控；
    2. 允许的最大 CER：
           baseline_CER + max_cer_increase
    3. 在满足 CER 约束的阈值中：
           优先 RR 最高；
           RR 相同时 CER 更低；
           再相同时 Pos 接受率更高。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_threshold = -1.000001
    baseline = evaluate_threshold(
        records=records,
        threshold=baseline_threshold,
        save_predictions=False,
    )

    baseline_cer = baseline["pos"]["CER"]
    max_allowed_cer = baseline_cer + max_cer_increase

    print("\n========== 阈值自动搜索 ==========")
    print(
        f"不门控基线 CER：{baseline_cer:.6f} "
        f"({baseline_cer:.4%})"
    )
    print(
        f"允许最大 CER 增量：{max_cer_increase:.6f} "
        f"（绝对值）"
    )
    print(
        f"允许最大 CER：{max_allowed_cer:.6f} "
        f"({max_allowed_cer:.4%})"
    )

    thresholds = make_thresholds(
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )

    search_rows: List[Dict[str, Any]] = []
    eligible_results: List[Dict[str, Any]] = []

    for index, threshold in enumerate(thresholds, start=1):
        result = evaluate_threshold(
            records=records,
            threshold=threshold,
            save_predictions=False,
        )

        row = {
            "threshold": threshold,
            "CER": result["pos"]["CER"],
            "RR": result["neg"]["RR"],
            "pos_accept_rate": result["pos"]["accept_rate"],
            "pos_gate_reject_count": result["pos"]["gate_reject_count"],
            "neg_accept_rate": result["neg"]["accept_rate"],
            "neg_gate_reject_rate": result["neg"]["gate_reject_rate"],
            "neg_correct_reject_count": result["neg"]["correct_reject_count"],
            "meets_cer_constraint": (
                result["pos"]["CER"] <= max_allowed_cer
            ),
        }
        search_rows.append(row)

        if row["meets_cer_constraint"]:
            eligible_results.append(result)

        if index % 20 == 0 or index == len(thresholds):
            print(
                f"阈值搜索：已完成 {index}/{len(thresholds)}，"
                f"当前 threshold={threshold:.4f}，"
                f"CER={row['CER']:.4%}，"
                f"RR={row['RR']:.4%}"
            )

    search_csv = output_dir / "threshold_search.csv"

    with open(search_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(search_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(search_rows)

    if not eligible_results:
        # 理论上至少很低阈值应满足，但仍增加保护。
        best_result = min(
            (
                evaluate_threshold(
                    records=records,
                    threshold=threshold,
                    save_predictions=False,
                )
                for threshold in thresholds
            ),
            key=lambda result: result["pos"]["CER"],
        )
        selection_reason = "没有阈值满足 CER 约束，改选 CER 最低阈值"
    else:
        best_result = max(
            eligible_results,
            key=lambda result: (
                result["neg"]["RR"],
                -result["pos"]["CER"],
                result["pos"]["accept_rate"],
            ),
        )
        selection_reason = (
            "在满足 CER 约束的阈值中，选择 RR 最高者"
        )

    best_threshold = float(best_result["threshold"])

    # 为最佳阈值重新保存逐条预测
    best_result = evaluate_threshold(
        records=records,
        threshold=best_threshold,
        output_dir=output_dir,
        save_predictions=True,
    )

    best_info = {
        "baseline_threshold": baseline_threshold,
        "baseline_CER": baseline_cer,
        "max_cer_increase": max_cer_increase,
        "max_allowed_CER": max_allowed_cer,
        "best_threshold": best_threshold,
        "selection_reason": selection_reason,
        "best_result": best_result,
        "threshold_search_csv": str(search_csv),
    }

    best_file = output_dir / "best_threshold.json"

    with open(best_file, "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    best_info["best_threshold_file"] = str(best_file)
    return best_info


# ============================================================
# 命令行
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DatasetA：短唤醒词声纹阈值门控 + Fun-ASR + 官方 CER/RR"
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="DatasetA 根目录，默认：项目根目录/datasetA",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "声纹余弦阈值。score>=threshold 接受，"
            "score<threshold 拒识"
        ),
    )
    parser.add_argument(
        "--search-threshold",
        action="store_true",
        help="自动扫描多个阈值并选择最佳阈值",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=-0.20,
        help="自动搜索的最小阈值",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.80,
        help="自动搜索的最大阈值",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="自动搜索的阈值步长",
    )
    parser.add_argument(
        "--max-cer-increase",
        type=float,
        default=0.09,
        help=(
            "相对不门控基线允许的 CER 最大绝对增量。"
            "例如 0.09 表示最多增加 9 个百分点"
        ),
    )

    parser.add_argument(
        "--asr-model",
        type=str,
        default=ASR_MODEL_NAME,
    )
    parser.add_argument(
        "--sv-model",
        type=str,
        default=SV_MODEL_NAME,
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default=DEFAULT_ASR_DEVICE,
    )
    parser.add_argument(
        "--sv-device",
        type=str,
        default=DEFAULT_SV_DEVICE,
    )
    parser.add_argument(
        "--hub",
        type=str,
        default="ms",
        choices=["ms", "hf"],
    )
    parser.add_argument(
        "--remote-code",
        type=Path,
        default=ROOT / "model.py",
        help="Fun-ASR-Nano 对应的本地 model.py",
    )

    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help=(
            "复用 output-dir/raw_model_outputs.jsonl，"
            "调整阈值时无需重新推理"
        ),
    )

    return parser


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    args = build_arg_parser().parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    pos_jsonl = data_root / "pos.jsonl"
    neg_jsonl = data_root / "neg.jsonl"
    cache_file = output_dir / "raw_model_outputs.jsonl"

    if not pos_jsonl.exists():
        raise FileNotFoundError(f"找不到：{pos_jsonl}")

    if not neg_jsonl.exists():
        raise FileNotFoundError(f"找不到：{neg_jsonl}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== DatasetA 配置 ==========")
    print(f"项目根目录：{ROOT}")
    print(f"数据目录：{data_root}")
    print(f"Pos 文件：{pos_jsonl}")
    print(f"Neg 文件：{neg_jsonl}")
    print(f"输出目录：{output_dir}")

    if args.reuse_cache:
        if not cache_file.exists():
            raise FileNotFoundError(
                f"指定了 --reuse-cache，但找不到缓存：{cache_file}"
            )

        records = load_raw_cache(cache_file)

    else:
        print("\n========== 读取 DatasetA ==========")
        pos_records = load_jsonl(pos_jsonl)
        neg_records = load_jsonl(neg_jsonl)

        records = prepare_records(
            data_root=data_root,
            pos_records=pos_records,
            neg_records=neg_records,
        )

        # 两个模型顺序加载，降低显存占用
        run_speaker_scoring(
            records=records,
            model_name=args.sv_model,
            device=args.sv_device,
            hub=args.hub,
        )

        run_asr(
            records=records,
            model_name=args.asr_model,
            device=args.asr_device,
            hub=args.hub,
            remote_code_path=args.remote_code,
        )

        save_raw_cache(
            records=records,
            cache_file=cache_file,
        )

    if args.search_threshold:
        best_info = search_best_threshold(
            records=records,
            output_dir=output_dir,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            max_cer_increase=args.max_cer_increase,
        )

        print_evaluation(
            best_info["best_result"],
            title="自动选择的最佳阈值结果",
        )

        print("\n----- 最佳阈值 -----")
        print(f"best_threshold：{best_info['best_threshold']:.6f}")
        print(f"选择规则：{best_info['selection_reason']}")
        print(
            f"阈值搜索表：{best_info['threshold_search_csv']}"
        )
        print(
            f"最佳阈值信息：{best_info['best_threshold_file']}"
        )

    else:
        result = evaluate_threshold(
            records=records,
            threshold=args.threshold,
            output_dir=output_dir,
            save_predictions=True,
        )

        print_evaluation(
            result,
            title="固定阈值测试结果",
        )

        print("\n----- 输出文件 -----")
        print(f"Pos 预测：{result['pos']['prediction_file']}")
        print(f"Neg 预测：{result['neg']['prediction_file']}")
        print(f"汇总文件：{result['summary_file']}")
        print(f"原始缓存：{cache_file}")


if __name__ == "__main__":
    main()