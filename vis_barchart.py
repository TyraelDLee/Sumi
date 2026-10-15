import matplotlib.pyplot as plt
import numpy as np

def plot_bar_chart(arr):
    x = range(len(arr))

    plt.figure(figsize=(6, 4))
    plt.bar(x, arr, color='skyblue')

    # 🔥 关键：缩放y轴范围
    min_val = min(arr)
    max_val = max(arr)
    margin = (max_val - min_val) * 0.2  # 留一点边界

    plt.ylim(min_val - margin, max_val + margin)

    plt.xticks(x)
    plt.title("Zoomed Bar Chart")

    for i, v in enumerate(arr):
        plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

    plt.show()

# plot_bar_chart([6.54, 6.48])


def borken_bar(label, value, x_label, y_label, font_size=26):


    plt.figure(figsize=(8, 6), dpi=300)
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    x = np.arange(len(label))

    # 每个柱使用不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(value)))  # 从 colormap 自动生成多色

    # 创建上下两个子图
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                            gridspec_kw={'height_ratios': [10, 1]})

    # 上部分：放大细节
    bars = ax_top.bar(x, value, color=colors, edgecolor='black', width=1)
    min_val = min(value)
    max_val = max(value)
    margin = (max_val - min_val) * 0.2  # 留一点边界

    ax_top.set_ylim(min_val - margin, max_val + margin)
    # ax_top.set_ylim(90.5, 92.1)
    ax_top.set_ylabel(y_label, fontsize=font_size)

    # 下部分：整体视角
    ax_bottom.bar(x, value, color=colors, edgecolor='black', width=1)
    ax_bottom.set_ylim(0, min_val/2)
    ax_bottom.set_xlabel(x_label, fontsize=font_size)

    # 🔹 添加断口标志（小斜线）
    d = .015  # 断口尺寸
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)  # 左侧斜线
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右侧斜线
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    for bar in bars:
        h = bar.get_height()
        # if h > 10:
        ax_top.text(
            bar.get_x() + bar.get_width() / 2, h,
            f"{h:.2f}", ha='center', va='bottom', fontsize=font_size, fontweight='bold'
        )

    # 美化
    plt.xticks(x, label, fontsize=font_size-2)
    ax_bottom.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

models = ['×16', '×2 w/ Upsampling']

# 两种评判方法的数值（你可以换成你的两行 8 列数据）
method_a = [6.54, 6.27]
borken_bar(models, method_a, 'Setting', 'Chamfer distance (↓)')