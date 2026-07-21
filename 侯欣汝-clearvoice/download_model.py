from modelscope import snapshot_download

# 从 ModelScope 下载 MossFormer2_SS_16K 模型
model_dir = snapshot_download('alibabasglab/MossFormer2_SS_16K')
print(f"模型已下载到: {model_dir}")