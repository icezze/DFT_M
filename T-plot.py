#!/usr/bin/env python3
#用于处理AIMD计算中的温度并按步长绘图，读取VASP的过程日志文件runlog（请根据时间计算环境更改文件名，可替换为OSZICAR）
import re, sys, pathlib, numpy as np, matplotlib.pyplot as plt

# --- 1. command-line input ---
log_file = sys.argv[1] if len(sys.argv) > 1 else "runlog"

# --- 2. extract temperatures ---
txt = pathlib.Path(log_file).read_text()
temps = [float(m) for m in re.findall(r"T=\s*([\d.]+)", txt)]
if not temps:
    sys.exit("No T= values found.")
steps = np.arange(1, len(temps) + 1)

# --- 3. save data ---
out_dat = "temperature.dat"
np.savetxt(out_dat, np.column_stack([steps, temps]), fmt="%d %.2f")
print(f"saved {len(temps)} records to {out_dat}")

# --- 4. plot ---
plt.figure(figsize=(6, 4))
plt.plot(steps, temps, lw=1.2, marker="o", markersize=2)
plt.xlabel("MD step")
plt.ylabel("Temperature (K)")
plt.title("Temperature evolution")
plt.grid(alpha=0.3)
plt.tight_layout()
out_png = "temperature.png"
plt.savefig(out_png, dpi=300)
print(f"figure saved to {out_png}")
plt.show()
