import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# 1. 数据（示例）
# 8 个模型
models = ['40%', '50%', '60%', '75%', '80%']

# 两种评判方法的数值（你可以换成你的两行 8 列数据）
method_a = [91.9105, 91.2220, 91.2220, 91.0499, 90.7057]

plt.figure(figsize=(8,6), dpi=300)
plt.rc('font', size=16)
plt.rc('axes', titlesize=16, labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
x = np.arange(len(models))

# 每个柱使用不同颜色
colors = plt.cm.tab10(np.linspace(0, 1, len(method_a)))  # 从 colormap 自动生成多色

# 创建上下两个子图
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(8,6),
                                       gridspec_kw={'height_ratios':[10,1]})

# 上部分：放大细节
bars = ax_top.bar(x, method_a, color=colors, edgecolor='black', width=0.6)
ax_top.set_ylim(90.5, 92.1)
ax_top.set_ylabel("Accuracy (%)" , fontsize=16)

# 下部分：整体视角
ax_bottom.bar(x, method_a, color=colors, edgecolor='black', width=0.6)
ax_bottom.set_ylim(0, 5)
ax_bottom.set_xlabel("Mask Ratio" , fontsize=16)

# 🔹 添加断口标志（小斜线）
d = .015  # 断口尺寸
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)            # 左侧斜线
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)      # 右侧斜线
kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

for bar in bars:
    h = bar.get_height()
    if h > 10:
        ax_top.text(
            bar.get_x() + bar.get_width()/2, h,
            f"{h:.2f}", ha='center', va='bottom', fontsize=16, fontweight='bold'
        )

# 美化
plt.xticks(x, models, fontsize=14)
ax_bottom.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,6), dpi=300)
plt.rc('font', size=16)
plt.rc('axes', titlesize=16, labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
labels = ['t=100, BS=1', 't=200, BS=1', 't=300, BS=1', 't=200, BS=32']
values = [8.22, 4.08, 1.89, 31.03]
x = np.arange(len(labels))

# 颜色：每个柱子不同
colors = plt.cm.tab10(np.linspace(0, 1, len(values)))

# 创建上下两个子图，共享 x 轴
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(8,6),
    gridspec_kw={'height_ratios': [1, 2]}
)

# -------- 下部分（主要显示普通值） --------
bars_bottom = ax_bottom.bar(x, values, color=colors, edgecolor='black', width=0.6)
ax_bottom.set_ylim(0, 10)  # 显示低值部分
ax_bottom.set_ylabel("items/sec" , fontsize=16)

# 在柱上显示数值（下部分）
for bar in bars_bottom:
    h = bar.get_height()
    if h < 10:  # 只标较小值
        ax_bottom.text(
            bar.get_x() + bar.get_width()/2, h + 0.3,
            f"{h:.2f}", ha='center', va='bottom', fontsize=16, fontweight='bold'
        )

# -------- 上部分（显示被截断的大值） --------
bars_top = ax_top.bar(x, values, color=colors, edgecolor='black', width=0.6)
ax_top.set_ylim(28, 33)  # 显示高值顶部部分
# ax_top.set_ylabel("items/sec (zoomed)")
# ax_top.set_title("Performance under Different Settings (Broken Axis)")

# 只在极值柱顶显示数值
for bar in bars_top:
    h = bar.get_height()
    if h > 10:
        ax_top.text(
            bar.get_x() + bar.get_width()/2, h + 0.2,
            f"{h:.2f}", ha='center', va='bottom', fontsize=16, fontweight='bold'
        )

# -------- 绘制断轴标志（两条小斜线） --------
d = .015
kwargs = dict(color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)

# -------- 其他美化 --------
plt.xticks(x, labels, rotation=25, ha='right')
ax_bottom.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.rc('font', size=14)
plt.show()
