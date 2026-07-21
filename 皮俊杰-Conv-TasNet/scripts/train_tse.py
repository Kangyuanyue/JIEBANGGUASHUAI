#!/usr/bin/env python3
"""
ConvTasNet TSE 训练: 基于 AISHELL-1 的说话人提取

用法:
  conda activate funasr
  python scripts/train_tse.py --dry-run    # 验证流程 
  python scripts/train_tse.py --epochs 10  # 开始训练
"""

import os, sys, json, time, argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================
# ConvTasNet 风格 TSE 模型 (兼容 Asteroid 预训练权重)
# ============================================================

class GlobalLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
    def forward(self, x):
        m = x.mean(dim=(1, 2), keepdim=True)
        s = x.std(dim=(1, 2), keepdim=True)
        return self.gamma * (x - m) / (s + 1e-8) + self.beta


class TCNBlock(nn.Module):
    """Conv-TasNet TCN block: 1x1→gLN→dwConv→gLN→PReLU→residual+skip"""
    def __init__(self, in_c, hid_c, out_c, ksize=3, d=1):
        super().__init__()
        p = (ksize - 1) * d // 2
        self.c1 = nn.Conv1d(in_c, hid_c, 1)
        self.n1 = GlobalLayerNorm(hid_c)
        self.dw = nn.Conv1d(hid_c, hid_c, ksize, padding=p, dilation=d, groups=hid_c)
        self.n2 = GlobalLayerNorm(hid_c)
        self.ac = nn.PReLU(hid_c)
        self.res = nn.Conv1d(hid_c, out_c, 1)
        self.skp = nn.Conv1d(hid_c, out_c, 1)
    def forward(self, x):
        x = self.ac(self.n1(self.c1(x)))
        x = self.dw(x); x = self.n2(x); x = self.ac(x)
        return self.res(x), self.skp(x)


class Repeat(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n=8, ksize=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            TCNBlock(in_c if i == 0 else out_c, hid_c, out_c, ksize, 2**i)
            for i in range(n)])
    def forward(self, x):
        s = None
        for b in self.blocks:
            r, sk = b(x)
            x = r
            s = sk if s is None else s + sk  # 非 in-place 累加
        return s


class TSEModel(nn.Module):
    """
    ConvTasNet 架构 + VE-VE 说话人提取 (~4.2M 参数)
    与 Asteroid ConvTasNet 兼容: encoder=Conv1D(1,512,16,8), TCN=3×8, decoder=ConvT1D(128,1)
    """
    def __init__(self, nf=512, bn=128, hd=512, sk=128, ks=16, st=8, nb=8, nr=3, fd=128, light=False):
        super().__init__()
        self.st = st
        if light:
            # 轻量模式: ~0.5M 参数, 适合 CPU 训练
            nf, bn, hd, sk, ks, st, nb, nr, fd = 128, 64, 128, 64, 20, 16, 4, 2, 64
        self.enc = nn.Conv1d(1, nf, ks, st, ks//2, bias=False)
        self.bn = nn.Sequential(nn.Conv1d(nf, bn, 1), GlobalLayerNorm(bn), nn.PReLU(bn))
        self.rpts = nn.ModuleList([Repeat(bn, hd, sk, nb, 3) for _ in range(nr)])
        self.tcn_out = nn.Conv1d(sk, bn, 1)
        self.spk = nn.Sequential(nn.Conv1d(nf, fd, 1), GlobalLayerNorm(fd), nn.PReLU(fd))
        self.cc = nn.Conv1d(bn + fd, bn, 1)
        self.dec = nn.ConvTranspose1d(bn, 1, ks, st, ks//2, bias=False)

    def enroll(self, audio):
        return self.spk(self.enc(audio).mean(dim=-1, keepdim=True))

    def extract(self, audio, emb):
        f = self.enc(audio)
        emb = emb.expand(-1, -1, f.shape[-1])
        f = self.cc(torch.cat([self.bn(f), emb], dim=1))
        s = None
        for r in self.rpts:
            skip = r(f)
            s = skip if s is None else s + skip  # 非 in-place 累加
            f = self.tcn_out(s)
        m = torch.sigmoid(self.dec(f))
        if m.shape[-1] != audio.shape[-1]:
            if m.shape[-1] > audio.shape[-1]: m = m[..., :audio.shape[-1]]
            else: m = torch.nn.functional.pad(m, (0, audio.shape[-1] - m.shape[-1]))
        return m * audio

    def forward(self, mix, enroll):
        return self.extract(mix, self.enroll(enroll))

# ──────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────

class TSEDataset(Dataset):
    """加载 TSE JSONL 数据"""
    def __init__(self, jsonl_path, max_enroll=16000, max_mix=48000, light=False):
        if light:
            max_enroll, max_mix = 8000, 24000  # 轻量: 更短的音频
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(l) for l in f if l.strip()]
        self.max_enroll = max_enroll
        self.max_mix = max_mix

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            import soundfile as sf
            enroll, _ = sf.read(r["enroll_path"])
            mixture, _ = sf.read(r["mixture_path"])
            target, _ = sf.read(r["target_path"])
        except:
            import librosa
            enroll, _ = librosa.load(r["enroll_path"], sr=16000)
            mixture, _ = librosa.load(r["mixture_path"], sr=16000)
            target, _ = librosa.load(r["target_path"], sr=16000)

        for arr in [enroll, mixture, target]:
            if arr.ndim > 1:
                arr = arr.mean(axis=1)

        # 截断
        enroll = enroll[:self.max_enroll].astype(np.float32)
        mixture = mixture[:self.max_mix].astype(np.float32)
        target = target[:self.max_mix].astype(np.float32)

        return enroll, mixture, target


def collate_fn(batch):
    """批量 padding"""
    enrolls, mixtures, targets = zip(*batch)
    max_e = max(e.shape[0] for e in enrolls)
    max_m = max(m.shape[0] for m in mixtures)

    e_pad = torch.zeros(len(batch), 1, max_e)
    m_pad = torch.zeros(len(batch), 1, max_m)
    t_pad = torch.zeros(len(batch), 1, max_m)

    for i, (e, m, t) in enumerate(batch):
        e_pad[i, 0, :e.shape[0]] = torch.from_numpy(e)
        m_pad[i, 0, :m.shape[0]] = torch.from_numpy(m)
        t_pad[i, 0, :t.shape[0]] = torch.from_numpy(t)

    return {"enroll": e_pad, "mixture": m_pad, "target": t_pad}


# ──────────────────────────────────────────────────────────────────────
# 训练
# ──────────────────────────────────────────────────────────────────────

def train_step(model, batch, optimizer, criterion, device):
    model.train()
    e, m, t = batch["enroll"].to(device), batch["mixture"].to(device), batch["target"].to(device)
    optimizer.zero_grad()
    enhanced = model(m, e)
    loss = criterion(enhanced, t)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss.item()


class SISDR(nn.Module):
    """SI-SDR 损失"""
    def forward(self, est, ref):
        if est.shape != ref.shape:
            min_l = min(est.shape[-1], ref.shape[-1])
            est, ref = est[..., :min_l], ref[..., :min_l]
        ref = ref / (torch.norm(ref, dim=-1, keepdim=True) + 1e-8)
        s_target = (est * ref).sum(dim=-1, keepdim=True) * ref
        e_noise = est - s_target
        si_sdr = 10 * torch.log10(
            (s_target**2).sum(dim=-1) / ((e_noise**2).sum(dim=-1) + 1e-8) + 1e-8)
        return -si_sdr.mean()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_jsonl", default=str(BASE / "data" / "tse_aishell" / "train.jsonl"))
    parser.add_argument("--val_jsonl", default=str(BASE / "data" / "tse_aishell" / "val.jsonl"))
    parser.add_argument("--save_dir", default=str(BASE / "outputs" / "tse_model"))
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="仅验证流程")
    parser.add_argument("--pretrained", type=str, default=str(BASE / "outputs" / "pretrained_convtasnet.pt"),
                        help="ConvTasNet 预训练权重路径 (不指定则从零训练)")
    parser.add_argument("--light", action="store_true", help="轻量模型 (~0.5M 参数, 适合 CPU)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 50)
    print("  TSE 训练: ConvTasNet + VE-VE")
    print("=" * 50)

    device = torch.device(args.device)
    print(f"  设备:     {device}")
    print(f"  数据:     {args.train_jsonl}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size}")

    # 检查数据
    if not os.path.exists(args.train_jsonl):
        print(f"[错误] 训练数据不存在: {args.train_jsonl}")
        print("  请先运行: python scripts/prepare_tse_aishell.py")
        sys.exit(1)

    # 模型
    is_light = args.light or args.device == "cpu"
    model = TSEModel(light=is_light)
    print(f"  参数量:   {count_params(model)/1e6:.2f}M")
    if is_light:
        print(f"  模式:     轻量 (CPU优化)")

    # 加载预训练权重 (可选)
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"  加载预训练: {args.pretrained}")
        try:
            ckpt = torch.load(args.pretrained, map_location="cpu")
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            # Asteroid → 我们的模型 键名映射
            def map_key(k):
                if k == "encoder.filterbank._filters": return "enc.weight"
                if k == "decoder.filterbank._filters": return "dec.weight"
                if k.startswith("masker.bottleneck.0."):
                    return "bn.1." + k.split(".", 2)[2]  # gamma/beta
                if k.startswith("masker.bottleneck.1."):
                    return "bn.0." + k.split(".", 2)[2]  # weight/bias
                if k == "masker.mask_conv.weight": return "tcn_out.weight"
                if k == "masker.mask_conv.bias": return "tcn_out.bias"
                # TCN blocks: masker.TCN.{repeat_idx}.{layer}.{param}
                import re
                m = re.match(r"masker\.TCN\.(\d+)\.(\w+)\.(\w+)", k)
                if m:
                    ri, layer, param = int(m.group(1)), m.group(2), m.group(3)
                    # Repeat/block index mapping: Asteroid has 8 blocks per repeat
                    bi = ri % 8
                    ri_actual = ri // 8
                    if layer == "shared_block":
                        sub = {"0": "c1", "1": None, "2": "n1", "3": "dw",
                               "4": None, "5": "n2"}
                        sub_key = sub.get(param.split("_")[0] if "_" in param else param, "")
                        # Handle gLN gamma/beta shape diff ([ch] vs [1,ch,1])
                        if param in ("gamma", "beta") and sub_key in ("n1", "n2"):
                            return f"rpts.{ri_actual}.blocks.{bi}.{sub_key}.{param}"
                        return f"rpts.{ri_actual}.blocks.{bi}.{sub_key}.{param}" if sub_key else None
                    elif layer in ("res_conv", "skip_conv"):
                        mapped = {"res_conv": "res", "skip_conv": "skp"}
                        return f"rpts.{ri_actual}.blocks.{bi}.{mapped[layer]}.{param}"
                return None

            model_sd = model.state_dict()
            matched = {}
            for pk in sd:
                mk = map_key(pk)
                if mk and mk in model_sd:
                    # Handle gLN shape diff: [ch] vs [1, ch, 1]
                    pv = sd[pk]
                    mv = model_sd[mk]
                    if len(pv.shape) == 1 and len(mv.shape) == 3:
                        pv = pv.reshape(mv.shape)
                    if pv.shape == mv.shape:
                        matched[mk] = pv

            if matched:
                model_sd.update(matched)
                model.load_state_dict(model_sd)
                print(f"    [OK] 匹配 {len(matched)}/{len(model_sd)} 层")
            else:
                print(f"    ! 无匹配层, 跳过预训练 (架构不同)")
        except Exception as e:
            print(f"    ! 加载失败: {e}")

    if args.dry_run:
        model = model.to(device)
        dummy = torch.randn(2, 1, 16000).to(device)
        embed = model.enroll(dummy)
        print(f"  enroll → {embed.shape}")
        out = model.extract(dummy, embed)
        print(f"  extract → {out.shape}")

        # 测试数据加载
        ds = TSEDataset(args.train_jsonl, light=is_light)
        dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn, shuffle=True)
        batch = next(iter(dl))
        print(f"  batch: enroll={batch['enroll'].shape}, mixture={batch['mixture'].shape}")

        # 测试前向
        criterion = SISDR()
        loss = criterion(model(batch["mixture"].to(device), batch["enroll"].to(device)),
                         batch["target"].to(device))
        print(f"  前向+损失: {loss.item():.4f}")
        print("\n[Dry-run 通过] 所有验证正常, 可开始训练。")
        return

    # 实际训练
    train_ds = TSEDataset(args.train_jsonl, light=is_light)
    val_ds = TSEDataset(args.val_jsonl, light=is_light) if os.path.exists(args.val_jsonl) else None
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn) if val_ds else None

    model = model.to(device)
    criterion = SISDR()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  开始训练 ({len(train_ds)} 条, {len(train_loader)} batch/epoch)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()
        for batch in train_loader:
            loss = train_step(model, batch, optimizer, criterion, device)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - t0

        # 验证
        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    e, m, t = batch["enroll"].to(device), batch["mixture"].to(device), batch["target"].to(device)
                    val_loss += criterion(model(m, e), t).item()
            val_loss /= len(val_loader)

        print(f"  Epoch {epoch:2d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"{'val='+f'{val_loss:.4f}' if val_loader else ''} | {epoch_time:.0f}s")

        # 每 5 epoch 保存
        if epoch % 5 == 0 or epoch == 1:
            torch.save(model.state_dict(), save_dir / f"model_ep{epoch}.pt")

    torch.save(model.state_dict(), save_dir / "model.pt")
    print(f"\n[完成] 模型保存于: {save_dir}")


if __name__ == "__main__":
    main()
