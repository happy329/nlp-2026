import sys
import math
import html
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline import (
    get_recommend_api,
    normalize_classification,
    predict_class,
    textrank_summarize,
)
from common.text_utils import load_stopwords, tokenize

st.set_page_config(page_title="中文新闻分类摘要与推荐系统", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1 {
        color: #2d3436;
        font-size: 2.75rem !important;
        font-weight: 800 !important;
        letter-spacing: 0;
    }
    p, label, .stMarkdown {
        font-size: 1.03rem;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 1rem;
        border: 2px solid #74b9ff;
        background: #ffffff;
        font-size: 1rem;
        line-height: 1.7;
        box-shadow: 0 8px 0 rgba(116, 185, 255, 0.14);
    }
    div[data-testid="stButton"] button {
        min-height: 3rem;
        border-radius: 1rem;
        border: 2px solid #74b9ff;
        font-weight: 700;
        box-shadow: 0 5px 0 rgba(45, 52, 54, 0.12);
    }
    div[data-testid="stMetric"] {
        background: #f0f8ff;
        border: 2px solid #74b9ff;
        border-radius: 1rem;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #74b9ff !important;
        border-radius: 1rem !important;
        background: rgba(255, 255, 255, 0.68);
        box-shadow: 0 8px 0 rgba(116, 185, 255, 0.12);
    }
    div[data-testid="stSidebar"] {
        border-right: 2px solid #74b9ff;
    }
    .keyword-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 0.75rem;
    }
    .keyword-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.58rem 0.95rem;
        border: 2px solid #74b9ff;
        border-radius: 999px;
        background: #f0f8ff;
        color: #2d3436;
        font-weight: 700;
        box-shadow: 0 4px 0 rgba(116, 185, 255, 0.18);
    }
    .keyword-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.45rem;
        height: 1.45rem;
        border-radius: 999px;
        background: #ff6b6b;
        color: #ffffff;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .summary-panel {
        margin-top: 0.8rem;
        padding: 1.35rem 1.55rem;
        border-radius: 1rem;
        border: 2px solid #74b9ff;
        background: #ffffff;
        box-shadow: 0 8px 0 rgba(116, 185, 255, 0.12);
    }
    .summary-title {
        margin-bottom: 0.75rem;
        color: #0984e3;
        font-size: 1rem;
        font-weight: 800;
    }
    .summary-text {
        color: #2d3436;
        font-size: 1.22rem;
        line-height: 1.9;
        font-weight: 700;
    }
    .summary-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin: 0.8rem 0 0.2rem;
    }
    .summary-stat {
        display: inline-flex;
        align-items: baseline;
        gap: 0.35rem;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        border: 1.5px solid #74b9ff;
        background: #f0f8ff;
        color: #636e72;
        font-size: 0.9rem;
        font-weight: 700;
    }
    .summary-stat strong {
        color: #2d3436;
        font-size: 1rem;
    }
    .recommend-title {
        padding: 0.78rem 0.9rem 0.78rem 1rem;
        border-left: 0.35rem solid #ff6b6b;
        border-radius: 0.75rem;
        background: rgba(240, 248, 255, 0.92);
        color: #263238;
        font-size: 1.04rem;
        line-height: 1.65;
        font-weight: 800;
        box-shadow: inset 0 0 0 1px rgba(116, 185, 255, 0.32);
        margin-bottom: 0.85rem;
    }
    details.original-expander {
        margin-top: 0.85rem;
        border: 2px solid #74b9ff;
        border-radius: 0.85rem;
        background: rgba(240, 248, 255, 0.76);
        overflow: hidden;
    }
    details.original-expander summary {
        cursor: pointer;
        padding: 0.72rem 0.9rem;
        color: #263238;
        font-weight: 800;
        list-style: none;
    }
    details.original-expander summary::-webkit-details-marker {
        display: none;
    }
    details.original-expander summary::before {
        content: "›";
        display: inline-block;
        margin-right: 0.58rem;
        color: #0984e3;
        font-size: 1.15rem;
        font-weight: 900;
        transition: transform 0.18s ease;
    }
    details.original-expander[open] summary::before {
        transform: rotate(90deg);
    }
    .original-content {
        padding: 0.9rem 1rem 1.05rem;
        border-top: 1.5px solid rgba(116, 185, 255, 0.55);
        background: rgba(255, 255, 255, 0.74);
        color: #2d3436;
        font-size: 0.98rem;
        line-height: 1.85;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("中文新闻分类摘要与推荐系统")
st.caption("新闻分类 · 关键词提取 · 自动摘要 · 相似新闻推荐")

# ---------- 侧边栏设置 ----------
with st.sidebar:
    st.header("设置")
    classification_model = st.radio(
        "分类模型",
        options=[
            ("bilstm_attention", "BiLSTM + Attention"),
            ("bilstm", "BiLSTM"),
            ("textcnn", "TextCNN"),
        ],
        format_func=lambda item: item[1],
    )[0]
    summary_method = st.radio(
        "摘要模型",
        options=[
            ("textrank", "TextRank 抽取式"),
            ("randeng", "Randeng-Pegasus 生成式"),
        ],
        format_func=lambda item: item[1],
    )[0]
    st.divider()
    top_k = st.slider("推荐数量", min_value=1, max_value=10, value=5)
    recommend_method = st.radio("推荐方法", ["sbert", "tfidf"], index=0)

# ---------- 输入 ----------
default_text = (
    "近日，某科技公司发布新一代人工智能芯片，主要面向大模型训练和推理场景。"
    "该芯片采用先进制程工艺，算力相比上一代提升数倍。"
    "业内专家表示，这将有力推动AI产业发展，提升国产芯片竞争力。"
    "多家互联网公司已表示将率先采用该芯片进行数据中心升级。"
)
input_text = st.text_area("请输入一篇新闻", value=default_text, height=180)

if "active_result" not in st.session_state:
    st.session_state.active_result = None


@st.cache_data
def load_news_lookup():
    news_path = PROJECT_ROOT / "data" / "processed" / "news.csv"
    if not news_path.exists():
        return {}
    df = pd.read_csv(news_path)
    return dict(zip(df["news_id"].astype(str), df["content"].astype(str)))


def get_news_content(item):
    news_id = str(item.get("news_id", ""))
    return load_news_lookup().get(news_id, item.get("content", ""))


def count_text_chars(text):
    return len("".join(str(text or "").split()))


def get_category_chart_path():
    candidates = [
        PROJECT_ROOT / "visualization" / "news_category_bar.png",
        PROJECT_ROOT / "results" / "figures" / "category_distribution.png",
        PROJECT_ROOT / "vis" / "news_category_bar.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def generate_recommendation_summary(content, method):
    content = str(content or "").strip()
    if not content:
        return "无摘要"

    if method == "textrank":
        return textrank_summarize(content)

    if method != "randeng":
        return f"未知摘要模型：{method}"

    from summarization.randeng_pegasus_summary import randeng_pegasus_summarize

    return randeng_pegasus_summarize(content)


def build_word_frequencies(text, max_words=50):
    stopwords = load_stopwords()
    words = tokenize(text, stopwords=stopwords, keep_single_char=False)
    return dict(Counter(words).most_common(max_words))


def create_wordcloud_figure(frequencies):
    font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    if not Path(font_path).exists():
        font_path = "/System/Library/Fonts/STHeiti Medium.ttc"

    wordcloud = WordCloud(
        font_path=font_path,
        width=1000,
        height=460,
        background_color="#fff9e6",
        colormap="Set2",
        max_words=50,
        prefer_horizontal=0.92,
        random_state=42,
        collocations=False,
        margin=8,
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#fff9e6")
    return fig


def format_hover_content(text, max_len=150, line_len=22):
    text = " ".join(str(text or "").split())
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return "<br>".join(
        html.escape(text)[index : index + line_len]
        for index in range(0, len(text), line_len)
    )


def create_similarity_graph(recs, source_text=""):
    labels = [item.get("label", "-") for item in recs]
    unique_labels = list(dict.fromkeys(labels))
    palette = [
        "#74b9ff",
        "#55efc4",
        "#ffeaa7",
        "#fab1a0",
        "#a29bfe",
        "#fd79a8",
        "#81ecec",
        "#e17055",
        "#00b894",
        "#0984e3",
    ]
    label_colors = {
        label: palette[index % len(palette)]
        for index, label in enumerate(unique_labels)
    }

    node_x = [0.0]
    node_y = [0.0]
    node_text = ["输入新闻"]
    node_hover = [f"输入新闻<br>{format_hover_content(source_text)}"]
    node_colors = ["#ff6b6b"]
    node_sizes = [56]
    radius = 2.1
    angles = [
        (2 * math.pi * index / max(len(recs), 1)) - math.pi / 2
        for index in range(len(recs))
    ]

    fig = go.Figure()

    for index, (item, angle) in enumerate(zip(recs, angles), start=1):
        score = float(item.get("score", 0) or 0)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        label = item.get("label", "-")
        news_id = item.get("news_id", "-")
        content = get_news_content(item)

        fig.add_trace(
            go.Scatter(
                x=[0, x],
                y=[0, y],
                mode="lines",
                line={
                    "color": f"rgba(116, 185, 255, {0.35 + min(score, 1) * 0.45})",
                    "width": 2 + score * 8,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

        mid_x = x * 0.44
        mid_y = y * 0.44
        fig.add_trace(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="markers+text",
                marker={
                    "size": 30,
                    "color": "#ffffff",
                    "line": {"color": "#dfe6e9", "width": 1},
                },
                text=[f"{score:.2f}"],
                textfont={"size": 11, "color": "#2d3436"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"R{index}<br>{label}")
        node_hover.append(
            f"推荐新闻 R{index}<br>ID：{news_id}<br>类别：{label}<br>"
            f"相似度：{score:.4f}<br><br>新闻内容：<br>{format_hover_content(content)}"
        )
        node_colors.append(label_colors.get(label, "#74b9ff"))
        node_sizes.append(42 + min(score, 1) * 18)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker={
                "size": node_sizes,
                "color": node_colors,
                "line": {"color": "#2d3436", "width": 2},
            },
            text=node_text,
            textposition="middle center",
            textfont={"size": 13, "color": "#2d3436", "family": "Arial"},
            hovertext=node_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={
            "text": "相似新闻关系图",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22, "color": "#2d3436"},
        },
        height=520,
        paper_bgcolor="#fff9e6",
        plot_bgcolor="#fff9e6",
        margin={"l": 12, "r": 12, "t": 72, "b": 24},
        xaxis={"visible": False, "range": [-2.9, 2.9]},
        yaxis={
            "visible": False,
            "range": [-2.55, 2.55],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#74b9ff",
            "font": {"color": "#2d3436", "size": 12},
            "align": "left",
        },
    )
    return fig


def has_text() -> bool:
    if input_text.strip():
        return True
    st.warning("请先输入一篇新闻")
    return False


def render_classification(cls):
    st.subheader("分类结果")
    label = cls.get("label", "-")
    label_id = cls.get("label_id", "-")
    probs = cls.get("probs", {})

    col_pred, col_bar = st.columns([1, 3])
    with col_pred:
        st.metric("预测类别", label, delta=f"ID: {label_id}")
    with col_bar:
        if probs:
            probs_df = (
                pd.DataFrame([{"类别": k, "概率": float(v)} for k, v in probs.items()])
                .sort_values("概率", ascending=True)
                .reset_index(drop=True)
            )
            st.bar_chart(probs_df.set_index("类别"))

    chart_path = get_category_chart_path()
    if chart_path:
        st.divider()
        st.subheader("新闻类别分布柱状图")
        st.image(str(chart_path), use_container_width=True)
    else:
        st.info("未找到 visualization 下的新闻类别分布图")


def render_keywords(keyword_result):
    st.subheader("关键词词云")
    frequencies = keyword_result.get("frequencies", {})
    if frequencies:
        fig = create_wordcloud_figure(frequencies)
        col_cloud, col_words = st.columns([3, 1])
        with col_cloud:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with col_words:
            st.caption("高频关键词")
            freq_df = pd.DataFrame(
                [{"关键词": word, "词频": count} for word, count in list(frequencies.items())[:10]]
            )
            st.dataframe(freq_df, hide_index=True, use_container_width=True)
    else:
        st.write("未提取到关键词")


def render_summary(summary):
    st.subheader("摘要结果")
    if summary.get("error"):
        st.error(summary["error"])
        return

    title = summary.get("title", "摘要")
    text = summary.get("text", "")
    original_chars = int(summary.get("original_chars", 0))
    summary_chars = int(summary.get("summary_chars", 0))
    compression_rate = summary.get("compression_rate", 0)

    if text:
        st.markdown(
            f"""
            <div class="summary-panel">
                <div class="summary-title">{html.escape(title)}</div>
                <div class="summary-text">{html.escape(text)}</div>
            </div>
            <div class="summary-stats">
                <div class="summary-stat">原文字数 <strong>{original_chars}</strong></div>
                <div class="summary-stat">摘要字数 <strong>{summary_chars}</strong></div>
                <div class="summary-stat">压缩率 <strong>{compression_rate:.1%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.write("未生成摘要")


def render_recommendations(recs, recommend_summary_method):
    st.subheader(f"相似新闻推荐")
    if recs:
        graph_fig = create_similarity_graph(recs, input_text)
        st.plotly_chart(graph_fig, use_container_width=True, config={"displayModeBar": False})

        for start in range(0, len(recs), 3):
            cols = st.columns(min(3, len(recs) - start))
            for col, item in zip(cols, recs[start : start + 3]):
                with col:
                    with st.container(border=True):
                        content = get_news_content(item)
                        try:
                            summary = generate_recommendation_summary(content, recommend_summary_method)
                        except Exception as exc:
                            summary = f"摘要暂时不可用：{exc}"
                        st.markdown(
                            f'<div class="recommend-title">{html.escape(summary)}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"类别：{item.get('label', '-')}｜ID：{item.get('news_id', '-')}")
                        st.progress(
                            float(item.get("score", 0)),
                            text=f"相似度 {item.get('score', 0):.4f}",
                        )
                        st.markdown(
                            f"""
                            <details class="original-expander">
                                <summary>展开原文</summary>
                                <div class="original-content">{html.escape(content)}</div>
                            </details>
                            """,
                            unsafe_allow_html=True,
                        )
    else:
        st.write("未找到相似新闻")


def generate_summary(text, method):
    original_chars = count_text_chars(text)
    if method == "textrank":
        summary_text = textrank_summarize(text)
        return {
            "title": "TextRank 抽取式摘要",
            "text": summary_text,
            "original_chars": original_chars,
            "summary_chars": count_text_chars(summary_text),
            "compression_rate": count_text_chars(summary_text) / original_chars if original_chars else 0,
        }

    try:
        if method == "randeng":
            from summarization.randeng_pegasus_summary import randeng_pegasus_summarize

            summary_text = randeng_pegasus_summarize(text)
            return {
                "title": "Randeng-Pegasus 摘要",
                "text": summary_text,
                "original_chars": original_chars,
                "summary_chars": count_text_chars(summary_text),
                "compression_rate": count_text_chars(summary_text) / original_chars if original_chars else 0,
            }
    except Exception as exc:
        return {
            "title": "摘要生成失败",
            "text": "",
            "error": f"当前摘要模型无法运行：{exc}",
            "original_chars": original_chars,
            "summary_chars": 0,
            "compression_rate": 0,
        }

    return {
        "title": "摘要生成失败",
        "text": "",
        "error": f"未知摘要模型：{method}",
        "original_chars": original_chars,
        "summary_chars": 0,
        "compression_rate": 0,
    }


st.write("")
btn_cls, btn_summary, btn_recommend, btn_keywords = st.columns(4)

with btn_cls:
    if st.button("分类", use_container_width=True):
        if has_text():
            with st.spinner("正在分类..."):
                st.session_state.active_result = {
                    "type": "classification",
                    "data": normalize_classification(
                        predict_class(input_text, model_name=classification_model, top_k=5)
                    ),
                }

with btn_summary:
    if st.button("生成摘要", use_container_width=True):
        if has_text():
            with st.spinner("正在生成摘要..."):
                st.session_state.active_result = {
                    "type": "summary",
                    "data": generate_summary(input_text, summary_method),
                }

with btn_recommend:
    if st.button("相似新闻推荐", use_container_width=True):
        if has_text():
            with st.spinner("正在推荐相似新闻..."):
                st.session_state.active_result = {
                    "type": "recommendations",
                    "data": get_recommend_api().recommend(
                        input_text,
                        label_filter=None,
                        top_k=top_k,
                        method=recommend_method,
                    ),
                }

with btn_keywords:
    if st.button("提取关键词", use_container_width=True):
        if has_text():
            with st.spinner("正在生成关键词词云..."):
                st.session_state.active_result = {
                    "type": "keywords",
                    "data": {
                        "frequencies": build_word_frequencies(input_text),
                    },
                }

if st.session_state.active_result:
    st.divider()
    with st.container(border=True):
        result_type = st.session_state.active_result["type"]
        result_data = st.session_state.active_result["data"]

        if result_type == "classification":
            render_classification(result_data)
        elif result_type == "summary":
            render_summary(result_data)
        elif result_type == "recommendations":
            render_recommendations(result_data, summary_method)
        elif result_type == "keywords":
            render_keywords(result_data)
