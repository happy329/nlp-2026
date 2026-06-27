"""
Streamlit 系统界面 app.py
接收用户输入新闻，并展示分类结果、关键词、自动摘要和相似新闻推荐结果
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline import run_pipeline

st.set_page_config(page_title="中文新闻分析系统", layout="wide")

st.title("中文新闻分析系统")
st.caption("新闻分类 · 关键词提取 · 自动摘要 · 相似新闻推荐")

# ---------- 侧边栏设置 ----------
with st.sidebar:
    st.header("设置")
    top_k = st.slider("推荐数量", min_value=1, max_value=10, value=5)
    recommend_method = st.radio("推荐方法", ["sbert", "tfidf"], index=0)
    st.divider()
    st.caption("SBERT：语义相似度更高，首次加载需联网\nTF-IDF：基于词频匹配，离线可用")

# ---------- 输入 ----------
default_text = (
    "近日，某科技公司发布新一代人工智能芯片，主要面向大模型训练和推理场景。"
    "该芯片采用先进制程工艺，算力相比上一代提升数倍。"
    "业内专家表示，这将有力推动AI产业发展，提升国产芯片竞争力。"
    "多家互联网公司已表示将率先采用该芯片进行数据中心升级。"
)
input_text = st.text_area("请输入新闻文本", value=default_text, height=180)

# ---------- 分析按钮 ----------
if st.button("开始分析", type="primary"):
    if not input_text.strip():
        st.warning("请先输入新闻文本")
    else:
        with st.spinner("分析中，请稍候..."):
            result = run_pipeline(
                input_text,
                top_k=top_k,
                recommend_method=recommend_method,
            )

        # ---------- 分类结果 ----------
        st.divider()
        st.subheader("📂 分类结果")
        cls = result.get("classification", {})
        label = cls.get("label", "-")
        label_id = cls.get("label_id", "-")
        probs = cls.get("probs", {})

        col_pred, col_bar = st.columns([1, 3])
        with col_pred:
            st.metric("预测类别", label, delta=f"ID: {label_id}")
        with col_bar:
            if probs:
                probs_df = (
                    pd.DataFrame(
                        [{"类别": k, "概率": float(v)} for k, v in probs.items()]
                    )
                    .sort_values("概率", ascending=True)
                    .reset_index(drop=True)
                )
                st.bar_chart(probs_df.set_index("类别"))

        # ---------- 关键词 ----------
        st.divider()
        st.subheader("🔑 关键词")
        keywords = result.get("keywords", [])
        if keywords:
            cols = st.columns(len(keywords))
            for col, kw in zip(cols, keywords):
                col.markdown(f":blue-badge[`{kw}`]")
        else:
            st.write("未提取到关键词")

        # ---------- 摘要 ----------
        st.divider()
        st.subheader("📝 摘要结果")
        summary = result.get("summary", {})

        # 抽取式摘要（TextRank，始终可用）
        textrank_summary = summary.get("textrank_summary", "")
        if textrank_summary:
            st.success(f"抽取式摘要（TextRank）：{textrank_summary}")

        # 生成式摘要（T5 / Pegasus）
        for key, title in [("t5_summary", "T5 生成式摘要"),
                           ("pegasus_summary", "Pegasus 生成式摘要")]:
            val = summary.get(key, "")
            if val:
                if val.startswith("["):
                    st.info(f"{title}：{val}")
                else:
                    st.success(f"{title}：{val}")

        # ---------- 相似推荐 ----------
        st.divider()
        st.subheader(f"📰 相似新闻推荐（Top-{top_k}）")
        recs = result.get("recommendations", [])
        if recs:
            cols = st.columns(min(len(recs), 3))
            for col, item in zip(cols, recs):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{item.get('title', '无标题')}**")
                        st.caption(f"类别：{item.get('label', '-')}｜ID：{item.get('news_id', '-')}")
                        st.progress(
                            float(item.get("score", 0)),
                            text=f"相似度 {item.get('score', 0):.4f}"
                        )
                        content = item.get("content", "")
                        if len(content) > 100:
                            content = content[:100] + "..."
                        st.text(content)
        else:
            st.write("未找到相似新闻")
