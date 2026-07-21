import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from clearvoice import ClearVoice

# ===== 配置 =====
INPUT_WAV = "cmd_50.wav"        # 你的混合音频文件名
OUTPUT_DIR = "output_separated"       # 输出目录

# ===== 开始分离 =====
print("=" * 50)
print("ClearVoice 语音分离 (MossFormer2_SS_16K)")
print("=" * 50)

# 检查输入文件
if not os.path.exists(INPUT_WAV):
    raise FileNotFoundError(f"找不到文件: {INPUT_WAV}，请把音频放在脚本同目录下")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输入文件: {INPUT_WAV}")
print(f"输出目录: {OUTPUT_DIR}")
print("正在加载模型（首次运行需下载，请耐心等待）...")

# 初始化分离模型
separator = ClearVoice(
    task="speech_separation",
    model_names=["MossFormer2_SS_16K"]
)

print("模型加载完成，开始分离...")

# 执行分离
separator(
    input_path=INPUT_WAV,
    online_write=True,           # 直接写入文件
    output_path=OUTPUT_DIR
)

# 列出结果
print("\n分离完成！生成的文件：")
for i, f in enumerate(sorted(os.listdir(OUTPUT_DIR))):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath) / 1024
    print(f"  [{i+1}] {f}  ({size:.1f} KB)")

print("\n✅ 完成！去 output_separated 文件夹里听效果吧。")