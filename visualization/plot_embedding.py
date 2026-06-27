"""
Sentence-BERT新闻向量降维可视化
展示不同类别新闻在语义空间中的分布情况
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import config as cfg


def plot_embedding_visualization(news_csv_path=None,
                                  sbert_embeddings_path=None,
                                  news_ids_path=None,
                                  save_path=None,
                                  method='tsne',
                                  sample_size=2000):
    """
    绘制SBERT向量降维可视化图

    Args:
        news_csv_path: news.csv路径
        sbert_embeddings_path: SBERT向量路径
        news_ids_path: 新闻ID路径
        save_path: 保存路径
        method: 降维方法 ('tsne' 或 'pca')
        sample_size: 采样数量（避免图太密集）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    if news_csv_path is None:
        news_csv_path = str(cfg.NEWS_CSV)
    if sbert_embeddings_path is None:
        sbert_embeddings_path = str(cfg.SBERT_EMBEDDINGS_NPY)
    if news_ids_path is None:
        news_ids_path = str(cfg.NEWS_IDS_JSON)

    df = pd.read_csv(news_csv_path)
    embeddings = np.load(sbert_embeddings_path)

    # 读取新闻ID映射
    news_ids = None
    if Path(news_ids_path).exists():
        import json
        with open(news_ids_path, 'r', encoding='utf-8') as f:
            news_ids = json.load(f)

    # 采样（如果数据量太大）
    if len(df) > sample_size:
        indices = np.random.choice(len(df), sample_size, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
        embeddings = embeddings[indices]

    print(f"加载新闻数据: {len(df)} 条")
    print(f"向量维度: {embeddings.shape}")

    # 降维
    print(f"使用 {method} 进行降维...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = 'Sentence-BERT新闻向量 t-SNE 可视化'
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = 'Sentence-BERT新闻向量 PCA 可视化'
        print(f"方差解释比例: {reducer.explained_variance_ratio_}")
    else:
        raise ValueError(f"不支持的降维方法: {method}")

    # 获取唯一类别
    categories = df['label'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    category_colors = dict(zip(categories, colors))

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))

    # 按类别绘制
    for category in categories:
        mask = df['label'] == category
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        ax.scatter(x, y, c=[category_colors[category]],
                  label=category, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('维度 1', fontsize=12)
    ax.set_ylabel('维度 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, markerscale=1.5)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # 保存
    if save_path is None:
        save_path = cfg.FIGURES_DIR / "embedding_visualization.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n嵌入可视化图已保存到: {save_path}")

    plt.close()
    return fig


def plot_embedding_by_category(sbert_embeddings_path=None,
                                news_csv_path=None,
                                save_path=None,
                                categories=None):
    """
    分别绘制每个类别的新闻向量分布（分面图）

    Args:
        sbert_embeddings_path: SBERT向量路径
        news_csv_path: 新闻CSV路径
        save_path: 保存路径
        categories: 要显示的类别列表（默认显示全部）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    if news_csv_path is None:
        news_csv_path = str(cfg.NEWS_CSV)
    if sbert_embeddings_path is None:
        sbert_embeddings_path = str(cfg.SBERT_EMBEDDINGS_NPY)

    df = pd.read_csv(news_csv_path)
    embeddings = np.load(sbert_embeddings_path)

    if categories is None:
        categories = df['label'].unique().tolist()

    print(f"类别: {categories}")

    # 使用PCA降维（更快）
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    # 创建分面子图
    n_cats = len(categories)
    n_cols = min(3, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_cats == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_cats))

    for idx, (category, ax) in enumerate(zip(categories, axes)):
        mask = df['label'] == category
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]

        ax.scatter(x, y, c=[colors[idx]], alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
        ax.set_title(f'类别: {category}', fontsize=12, fontweight='bold')
        ax.set_xlabel('PCA 维度 1', fontsize=10)
        ax.set_ylabel('PCA 维度 2', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')

    # 隐藏多余子图
    for idx in range(n_cats, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path is None:
        save_path = cfg.FIGURES_DIR / "embedding_by_category.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n分类别嵌入图已保存到: {save_path}")

    plt.close()
    return fig


if __name__ == "__main__":
    print("绘制SBERT嵌入可视化图...")

    # 检查索引文件是否存在
    if not cfg.SBERT_EMBEDDINGS_NPY.exists():
        print(f"错误: SBERT向量文件不存在: {cfg.SBERT_EMBEDDINGS_NPY}")
        print("请先运行 recommendation/build_index.py 构建索引")
    else:
        plot_embedding_visualization(method='tsne', sample_size=2000)
        plot_embedding_by_category()

    print("\n完成！")
