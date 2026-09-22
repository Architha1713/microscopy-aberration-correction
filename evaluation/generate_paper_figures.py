import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['figure.dpi'] = 300

RESULTS_DIR = r"D:\Projects\Microscopy_AI_Project\results"

# ============================================================
# FIGURE: PSNR / SSIM Before vs. After Correction (Run 8 - FINAL)
# ============================================================
psnr_before, psnr_after = 24.51, 26.51
ssim_before, ssim_after = 0.629, 0.707

psnr_pct = ((psnr_after - psnr_before) / psnr_before) * 100
ssim_pct = ((ssim_after - ssim_before) / ssim_before) * 100

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4))

bars1 = axes[0].bar(["Degraded\nInput", "Model\nOutput"], [psnr_before, psnr_after],
                     color=["#9CA3AF", "#1A8F84"], width=0.5)
axes[0].set_ylabel("PSNR (dB)")
axes[0].set_ylim(0, max(psnr_before, psnr_after) * 1.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
for bar in bars1:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha='center', fontsize=10)
axes[0].annotate(f"+{psnr_pct:.1f}%", xy=(1, psnr_after), xytext=(0.5, psnr_after + 4),
                  fontsize=10, color="#1A8F84", fontweight='bold', ha='center',
                  arrowprops=dict(arrowstyle="->", color="#1A8F84", lw=1.2))

bars2 = axes[1].bar(["Degraded\nInput", "Model\nOutput"], [ssim_before, ssim_after],
                     color=["#9CA3AF", "#1A8F84"], width=0.5)
axes[1].set_ylabel("SSIM")
axes[1].set_ylim(0, 1.0)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
for bar in bars2:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}", ha='center', fontsize=10)
axes[1].annotate(f"+{ssim_pct:.1f}%", xy=(1, ssim_after), xytext=(0.5, ssim_after + 0.15),
                  fontsize=10, color="#1A8F84", fontweight='bold', ha='center',
                  arrowprops=dict(arrowstyle="->", color="#1A8F84", lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figure_metrics_comparison.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, "figure_metrics_comparison.tiff"), dpi=300, bbox_inches='tight')
print("Saved figure_metrics_comparison (.png/.tiff) with FINAL Run 8 numbers")