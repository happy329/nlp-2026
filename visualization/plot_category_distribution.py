"""
类别分布可视化
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config as cfg


def plot_category_distribution():
    df = pd.read_csv(cfg.NEWS_CSV)
    counts = df["label"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("新闻类别分布")
    plt.xlabel("类别")
    plt.ylabel("数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "category_distribution.png", dpi=200)
    plt.close()
    print(f"已保存: {cfg.FIGURES_DIR / 'category_distribution.png'}")
