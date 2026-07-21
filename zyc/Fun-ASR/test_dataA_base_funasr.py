import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from funasr import AutoModel


# ============================================================
# 配置
# ============================================================
ROOT = Path(__file__).resolve().parent

# 数据集目录
DATA_ROOT = ROOT / "datasetA"

POS_JSONL = DATA_ROOT / "pos.jsonl"
NEG_JSONL = DATA_ROOT / "neg.jsonl"

# 输出目录
OUTPUT_DIR = ROOT / "outputs" / "base_funasr_dataA"

# Fun-ASR 模型
MODEL_NAME = "FunAudioLLM/Fun-ASR-Nano-2512"

# 没有 NVIDIA 显卡时改成 "cpu"
DEVICE = "cuda:0"

# 本地自定义模型代码。
# 如果 ROOT/model.py 不存在，则不传 remote_code 参数。
LOCAL_REMOTE_CODE = ROOT / "model.py"


# ============================================================
# 数据读取
# ============================================================
def repair_key(key: Any) -> Any:
    """
    尝试修复中文字段名乱码。

    例如某些 UTF-8 中文经过错误的 GBK 解码后，
    可以通过 encode("gbk").decode("utf-8") 尝试恢复。

    正常中文字段会原样返回。
    """
    if not isinstance(key, str):
        return key

    try:
        repaired = key.encode("gbk").decode("utf-8")
        return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        return key


def get_field(item: Dict[str, Any], field_name: str) -> Any:
    """
    获取 JSON 字段。

    同时兼容：
    1. 正常中文字段名；
    2. 因编码问题产生的乱码字段名。
    """
    for key, value in item.items():
        if key == field_name:
            return value

        if repair_key(key) == field_name:
            return value

    return None


def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    读取 JSONL 文件。

    依次尝试：
    1. utf-8-sig
    2. utf-8
    3. gb18030
    """
    encodings = ["utf-8-sig", "utf-8", "gb18030"]
    last_unicode_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            records: List[Dict[str, Any]] = []

            with open(jsonl_path, "r", encoding=encoding) as file:
                for line_no, line in enumerate(file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{jsonl_path.name} 第 {line_no} 行 JSON 格式错误："
                            f"{error}"
                        ) from error

                    if not isinstance(item, dict):
                        raise ValueError(
                            f"{jsonl_path.name} 第 {line_no} 行不是 JSON 对象："
                            f"{type(item).__name__}"
                        )

                    records.append(item)

            print(
                f"读取成功：{jsonl_path.name}，"
                f"编码={encoding}，样本数={len(records)}"
            )
            return records

        except UnicodeDecodeError as error:
            last_unicode_error = error

    raise RuntimeError(
        f"无法读取文件：{jsonl_path}\n"
        f"最后一次编码错误：{last_unicode_error}"
    )


# ============================================================
# 官方文本归一化
# ============================================================
def normalize_text(text: Any) -> str:
    """
    按比赛官方规则进行 ASR 文本归一化。

    处理步骤：
    1. None 转为空字符串；
    2. Unicode NFKC 规范化；
    3. 英文字母转小写；
    4. 去除前后空白；
    5. 删除所有空白字符；
    6. 删除 Unicode 类别为 P* 的标点字符。

    注意：
    - 仅删除 Unicode 标点类别 P*；
    - 不擅自删除汉字、英文字母、数字及其他非标点符号。
    """
    if text is None:
        return ""

    text = str(text)

    # 全角、半角等形式统一
    text = unicodedata.normalize("NFKC", text)

    # 英文字母统一为小写
    text = text.lower()

    # 删除前后空白
    text = text.strip()

    normalized_chars: List[str] = []

    for char in text:
        # 删除所有 Unicode 空白字符
        if char.isspace():
            continue

        # 删除所有 Unicode 标点字符：
        # Pc、Pd、Pe、Pf、Pi、Po、Ps 等类别
        if unicodedata.category(char).startswith("P"):
            continue

        normalized_chars.append(char)

    return "".join(normalized_chars)


# ============================================================
# Levenshtein 编辑距离及 S/I/D 统计
# ============================================================
def levenshtein_details(
    reference: str,
    hypothesis: str,
) -> Dict[str, int]:
    """
    计算字符级 Levenshtein 编辑距离，并统计：

    S：Substitution，替换错误
    I：Insertion，插入错误
    D：Deletion，删除错误

    参数：
    reference:
        规范化后的参考文本，即 Ground Truth。

    hypothesis:
        规范化后的预测文本，即 ASR Hypothesis。

    返回：
    {
        "substitutions": S,
        "insertions": I,
        "deletions": D,
        "errors": S + I + D
    }

    说明：
    当存在多种相同最小编辑距离的对齐方式时，
    S/I/D 的具体拆分可能存在多种合法结果，但：
        S + I + D
    始终等于标准 Levenshtein 编辑距离，因此 CER 不受影响。
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    # dp[i][j] 表示：
    # reference[:i] 转换成 hypothesis[:j] 的最小编辑距离
    dp = [
        [0 for _ in range(hyp_len + 1)]
        for _ in range(ref_len + 1)
    ]

    # operation[i][j] 记录到达该位置使用的操作：
    # E：Equal
    # S：Substitution
    # I：Insertion
    # D：Deletion
    operation = [
        ["" for _ in range(hyp_len + 1)]
        for _ in range(ref_len + 1)
    ]

    # reference 非空、hypothesis 为空：
    # 需要删除 reference 中的字符
    for i in range(1, ref_len + 1):
        dp[i][0] = i
        operation[i][0] = "D"

    # reference 为空、hypothesis 非空：
    # hypothesis 中的字符均为插入
    for j in range(1, hyp_len + 1):
        dp[0][j] = j
        operation[0][j] = "I"

    # 动态规划
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            ref_char = reference[i - 1]
            hyp_char = hypothesis[j - 1]

            if ref_char == hyp_char:
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

            # 相同代价时优先级：
            # 替换 > 删除 > 插入
            #
            # 该优先级只影响存在多解时 S/I/D 的拆分，
            # 不影响最终编辑距离和 CER。
            if best_cost == substitution_cost:
                operation[i][j] = "S"
            elif best_cost == deletion_cost:
                operation[i][j] = "D"
            else:
                operation[i][j] = "I"

    # 回溯统计 S、I、D
    substitutions = 0
    insertions = 0
    deletions = 0

    i = ref_len
    j = hyp_len

    while i > 0 or j > 0:
        current_operation = operation[i][j]

        if current_operation == "E":
            i -= 1
            j -= 1

        elif current_operation == "S":
            substitutions += 1
            i -= 1
            j -= 1

        elif current_operation == "D":
            deletions += 1
            i -= 1

        elif current_operation == "I":
            insertions += 1
            j -= 1

        else:
            raise RuntimeError(
                f"编辑距离回溯失败：i={i}, j={j}, "
                f"operation={current_operation!r}"
            )

    total_errors = substitutions + insertions + deletions

    # 内部一致性检查
    if total_errors != dp[ref_len][hyp_len]:
        raise RuntimeError(
            "编辑距离统计异常："
            f"S+I+D={total_errors}，"
            f"DP距离={dp[ref_len][hyp_len]}"
        )

    return {
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "errors": total_errors,
    }


# ============================================================
# CER 指标类
# ============================================================
class CERMetric:
    """
    Character Error Rate 字符错误率。

    官方公式：

        CER = (S + I + D) / N

    其中：
    S：替换错误数
    I：插入错误数
    D：删除错误数
    N：全部参考文本的总字符数

    重要：
    比赛整体 CER 应按全数据集累计后计算：

        总错误数 / 总参考字符数

    而不是先计算每条 CER，再求算术平均。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """清空所有累计统计。"""
        self.total_chars = 0
        self.total_errors = 0

        self.total_substitutions = 0
        self.total_insertions = 0
        self.total_deletions = 0

        self.per_sample_results: List[Dict[str, Any]] = []

    def update(
        self,
        preds: Union[str, List[str]],
        targets: Union[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        加入一条或多条预测结果。

        参数：
        preds:
            ASR 预测文本，或者预测文本列表。

        targets:
            参考文本，或者参考文本列表。

        返回：
            本次新增的单样本统计列表。
        """
        if isinstance(preds, str):
            preds = [preds]

        if isinstance(targets, str):
            targets = [targets]

        if len(preds) != len(targets):
            raise ValueError(
                "preds 和 targets 长度必须一致："
                f"{len(preds)} != {len(targets)}"
            )

        current_results: List[Dict[str, Any]] = []

        for pred, target in zip(preds, targets):
            original_pred = "" if pred is None else str(pred)
            original_target = "" if target is None else str(target)

            normalized_pred = normalize_text(original_pred)
            normalized_target = normalize_text(original_target)

            details = levenshtein_details(
                reference=normalized_target,
                hypothesis=normalized_pred,
            )

            substitutions = details["substitutions"]
            insertions = details["insertions"]
            deletions = details["deletions"]
            errors = details["errors"]

            target_chars = len(normalized_target)

            # 按官方参考代码处理空目标情况
            if target_chars == 0:
                sample_cer = 0.0 if errors == 0 else 1.0
            else:
                sample_cer = errors / target_chars

            self.total_substitutions += substitutions
            self.total_insertions += insertions
            self.total_deletions += deletions

            self.total_errors += errors
            self.total_chars += target_chars

            sample_result = {
                "orig_pred": original_pred,
                "orig_target": original_target,
                "norm_pred": normalized_pred,
                "norm_target": normalized_target,
                "substitutions": substitutions,
                "insertions": insertions,
                "deletions": deletions,
                "errors": errors,
                "target_chars": target_chars,
                "cer": sample_cer,
            }

            self.per_sample_results.append(sample_result)
            current_results.append(sample_result)

        return current_results

    def compute(self) -> Dict[str, Any]:
        """
        计算整个数据集的 CER。

        不是单句 CER 平均值，而是：

            所有样本总错误数 / 所有参考文本总字符数
        """
        if self.total_chars == 0:
            overall_cer = 0.0 if self.total_errors == 0 else 1.0
        else:
            overall_cer = self.total_errors / self.total_chars

        return {
            "cer": overall_cer,
            "total_errors": self.total_errors,
            "total_chars": self.total_chars,
            "total_substitutions": self.total_substitutions,
            "total_insertions": self.total_insertions,
            "total_deletions": self.total_deletions,
            "per_sample": self.per_sample_results,
        }


# ============================================================
# Fun-ASR 推理
# ============================================================
def recognize_one(
    model: AutoModel,
    wav_path: Path,
) -> Dict[str, Any]:
    """
    对单个音频执行 Fun-ASR 推理。

    返回：
    {
        "success": 是否推理成功,
        "text": 识别文本,
        "error": 错误信息
    }

    注意：
    推理失败和模型正常输出空字符串是两个不同概念。
    """
    try:
        result = model.generate(
            input=[str(wav_path)],
            cache={},
            batch_size=1,
            language="中文",
            itn=False,
        )

        if not result:
            return {
                "success": True,
                "text": "",
                "error": None,
            }

        first_result = result[0]

        if isinstance(first_result, dict):
            text = first_result.get("text", "")
        else:
            text = str(first_result)

        if text is None:
            text = ""

        return {
            "success": True,
            "text": str(text).strip(),
            "error": None,
        }

    except Exception as error:
        return {
            "success": False,
            "text": "",
            "error": f"{type(error).__name__}: {error}",
        }


# ============================================================
# 数据集测试
# ============================================================
def run_split(
    model: AutoModel,
    split_name: str,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    测试 pos 或 neg 数据集。

    Pos：
    - 计算标准 CER；
    - 分别统计 S、I、D；
    - 推理失败时预测按空字符串处理，因此会形成删除错误。

    Neg：
    - 模型正常推理且规范化结果为空，视为拒识成功；
    - 推理异常不算拒识成功；
    - RR = 正确拒识数 / 有效负样本数。
    """
    if split_name not in {"pos", "neg"}:
        raise ValueError(
            f"split_name 只能是 'pos' 或 'neg'，当前为：{split_name}"
        )

    output_file = OUTPUT_DIR / f"{split_name}_prediction.jsonl"

    cer_metric = CERMetric()

    valid_count = 0
    missing_count = 0
    inference_failed_count = 0

    neg_correct_reject = 0
    neg_nonempty_output_count = 0

    with open(output_file, "w", encoding="utf-8") as output_stream:
        for index, item in enumerate(records, start=1):
            cmd_rel_path = get_field(item, "识别音频")
            ref_text = get_field(item, "识别文本")

            # ------------------------------------------------
            # 检查命令音频字段
            # ------------------------------------------------
            if cmd_rel_path is None or str(cmd_rel_path).strip() == "":
                print(
                    f"[跳过] {split_name} 第 {index} 条"
                    f"没有“识别音频”字段"
                )
                missing_count += 1
                continue

            # Pos 必须有参考文本
            if split_name == "pos" and ref_text is None:
                print(
                    f"[跳过] pos 第 {index} 条"
                    f"没有“识别文本”字段"
                )
                missing_count += 1
                continue

            wav_path = (DATA_ROOT / str(cmd_rel_path)).resolve()

            # ------------------------------------------------
            # 检查音频文件
            # ------------------------------------------------
            if not wav_path.exists():
                print(f"[跳过] 音频不存在：{wav_path}")
                missing_count += 1
                continue

            # ------------------------------------------------
            # 执行 ASR
            # ------------------------------------------------
            recognition = recognize_one(
                model=model,
                wav_path=wav_path,
            )

            inference_success = bool(recognition["success"])
            pred_text = str(recognition["text"])
            inference_error = recognition["error"]

            if not inference_success:
                inference_failed_count += 1

                print(
                    f"[推理失败] {split_name} 第 {index} 条："
                    f"{wav_path.name}"
                )
                print(f"错误：{inference_error}")

            output_item: Dict[str, Any] = {
                "index": index,
                "id": item.get("id"),
                "唤醒音频": get_field(item, "唤醒音频"),
                "唤醒文本": get_field(item, "唤醒文本"),
                "识别音频": str(cmd_rel_path),
                "音频绝对路径": str(wav_path),
                "参考文本": ref_text,
                "预测文本": pred_text,
                "推理是否成功": inference_success,
                "推理错误": inference_error,
            }

            # ------------------------------------------------
            # Pos：计算标准 CER
            # ------------------------------------------------
            if split_name == "pos":
                sample_result = cer_metric.update(
                    preds=pred_text,
                    targets="" if ref_text is None else str(ref_text),
                )[0]

                output_item.update({
                    "规范化参考文本": sample_result["norm_target"],
                    "规范化预测文本": sample_result["norm_pred"],
                    "参考字符数N": sample_result["target_chars"],
                    "替换错误S": sample_result["substitutions"],
                    "插入错误I": sample_result["insertions"],
                    "删除错误D": sample_result["deletions"],
                    "字符错误总数": sample_result["errors"],
                    "单句CER": round(sample_result["cer"], 8),
                })

            # ------------------------------------------------
            # Neg：计算拒识率 RR
            # ------------------------------------------------
            else:
                normalized_pred = normalize_text(pred_text)

                # 只有推理正常完成且输出为空，才算真正拒识成功。
                # 推理异常不能作为拒识成功。
                is_correct_reject = (
                    inference_success
                    and normalized_pred == ""
                )

                if is_correct_reject:
                    neg_correct_reject += 1
                else:
                    neg_nonempty_output_count += 1

                output_item.update({
                    "规范化预测文本": normalized_pred,
                    "是否正确拒识": is_correct_reject,
                })

            output_stream.write(
                json.dumps(output_item, ensure_ascii=False) + "\n"
            )

            valid_count += 1

            if index % 20 == 0 or index == len(records):
                print(
                    f"{split_name}: 已扫描 {index}/{len(records)}，"
                    f"有效样本={valid_count}，"
                    f"缺失样本={missing_count}，"
                    f"推理失败={inference_failed_count}"
                )

    result: Dict[str, Any] = {
        "split": split_name,
        "input_count": len(records),
        "valid_count": valid_count,
        "missing_count": missing_count,
        "inference_failed_count": inference_failed_count,
        "prediction_file": str(output_file),
    }

    # Pos 整体 CER
    if split_name == "pos":
        cer_result = cer_metric.compute()

        result.update({
            "total_ref_chars": cer_result["total_chars"],
            "total_char_errors": cer_result["total_errors"],
            "total_substitutions": cer_result["total_substitutions"],
            "total_insertions": cer_result["total_insertions"],
            "total_deletions": cer_result["total_deletions"],
            "CER": cer_result["cer"],
        })

    # Neg 整体 RR
    else:
        if valid_count == 0:
            rr = 0.0
        else:
            rr = neg_correct_reject / valid_count

        result.update({
            "correct_reject_count": neg_correct_reject,
            "incorrect_reject_count": neg_nonempty_output_count,
            "RR": rr,
        })

    return result


# ============================================================
# 加载模型
# ============================================================
def load_model() -> AutoModel:
    """
    加载 Fun-ASR 模型。

    如果当前代码目录下存在 model.py，则作为本地 remote_code 使用；
    否则由模型仓库自行处理远程代码。
    """
    model_kwargs: Dict[str, Any] = {
        "model": MODEL_NAME,
        "trust_remote_code": True,
        "device": DEVICE,
        "hub": "ms",
    }

    if LOCAL_REMOTE_CODE.exists():
        model_kwargs["remote_code"] = str(LOCAL_REMOTE_CODE)
        print(f"使用本地 remote_code：{LOCAL_REMOTE_CODE}")
    else:
        print(
            f"未发现本地 model.py：{LOCAL_REMOTE_CODE}\n"
            "将不传入 remote_code 文件路径。"
        )

    model = AutoModel(**model_kwargs)
    return model


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # 检查数据文件
    # --------------------------------------------------------
    if not POS_JSONL.exists():
        raise FileNotFoundError(f"找不到 Pos 文件：{POS_JSONL}")

    if not NEG_JSONL.exists():
        raise FileNotFoundError(f"找不到 Neg 文件：{NEG_JSONL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------
    print("\n========== 读取 DatasetA ==========")

    pos_records = load_jsonl(POS_JSONL)
    neg_records = load_jsonl(NEG_JSONL)

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------
    print("\n========== 加载 Fun-ASR 官方权重 ==========")

    model = load_model()

    # --------------------------------------------------------
    # 测试 Pos
    # --------------------------------------------------------
    print("\n========== 测试 Pos：官方 CER ==========")

    pos_result = run_split(
        model=model,
        split_name="pos",
        records=pos_records,
    )

    # --------------------------------------------------------
    # 测试 Neg
    # --------------------------------------------------------
    print("\n========== 测试 Neg：RR ==========")

    neg_result = run_split(
        model=model,
        split_name="neg",
        records=neg_records,
    )

    # --------------------------------------------------------
    # 保存汇总结果
    # --------------------------------------------------------
    summary_file = OUTPUT_DIR / "summary.json"

    summary = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "data_root": str(DATA_ROOT),
        "cer_formula": "CER = (S + I + D) / N",
        "pos": pos_result,
        "neg": neg_result,
    }

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 打印最终结果
    # --------------------------------------------------------
    print("\n========== 最终结果 ==========")

    print("\n----- Pos CER -----")
    print(
        f"替换错误 S："
        f"{pos_result['total_substitutions']}"
    )
    print(
        f"插入错误 I："
        f"{pos_result['total_insertions']}"
    )
    print(
        f"删除错误 D："
        f"{pos_result['total_deletions']}"
    )
    print(
        f"总错误数 S+I+D："
        f"{pos_result['total_char_errors']}"
    )
    print(
        f"总参考字符数 N："
        f"{pos_result['total_ref_chars']}"
    )
    print(
        f"Pos CER："
        f"{pos_result['CER']:.6f} "
        f"({pos_result['CER']:.4%})"
    )

    print("\n----- Neg RR -----")
    print(
        f"有效负样本数："
        f"{neg_result['valid_count']}"
    )
    print(
        f"正确拒识数："
        f"{neg_result['correct_reject_count']}"
    )
    print(
        f"推理失败数："
        f"{neg_result['inference_failed_count']}"
    )
    print(
        f"Neg RR："
        f"{neg_result['RR']:.6f} "
        f"({neg_result['RR']:.4%})"
    )

    print("\n----- 输出位置 -----")
    print(f"预测文件目录：{OUTPUT_DIR}")
    print(f"汇总结果文件：{summary_file}")


if __name__ == "__main__":
    main()