"""
Top-K 推荐相似度可视化
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import config as cfg


def plot_similarity_topk(recommend_examples_path=None, save_path=None):
    recommend_examples_path = recommend_examples_path or cfg.RECOMMEND_EXAMPLES_CSV
    save_path = save_path or (cfg.FIGURES_DIR / "similarity_topk.png")

    if not Path(recommend_examples_path).exists():
        print("未找到推荐样例，跳过")
        return

    df = pd.read_csv(recommend_examples_path)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="news_id", y="score", hue="label")
    plt.title("Top-K 相似度")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")
