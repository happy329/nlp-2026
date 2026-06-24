import os

import matplotlib.pyplot as plt
import pandas as pd


news_file = "data/processed/news.csv"
output_fig = "vis/news_category_bar.png"


def set_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_category_bar():
    if not os.path.exists(news_file):
        print("没有找到新闻数据文件:", news_file)
        return

    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    set_chinese_font()

    df = pd.read_csv(news_file)
    counts = df["label"].value_counts().sort_values(ascending=False)
    total = int(counts.sum())

    colors = [
        "#3B82F6",
        "#10B981",
        "#F59E0B",
        "#EF4444",
        "#8B5CF6",
        "#06B6D4",
        "#F97316",
        "#14B8A6",
        "#6366F1",
        "#84CC16",
        "#EC4899",
        "#64748B",
        "#A855F7",
    ]

    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=150)
    bars = ax.bar(counts.index, counts.values, color=colors[: len(counts)], width=0.68)

    ax.set_title("新闻类别数量分布", fontsize=20, fontweight="bold", pad=18)
    ax.set_xlabel("新闻类别", fontsize=13, labelpad=10)
    ax.set_ylabel("新闻数量", fontsize=13, labelpad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    max_count = int(counts.max())
    ax.set_ylim(0, max_count * 1.14)

    for bar, value in zip(bars, counts.values):
        percent = value / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_count * 0.015,
            f"{value}\n{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)

    fig.text(
        0.99,
        0.02,
        f"数据来源: data/processed/news.csv    总样本数: {total}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#64748B",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print("新闻分类柱状图已保存到:", output_fig)
    print("总样本数:", total)
    print(counts)


if __name__ == "__main__":
    plot_category_bar()
