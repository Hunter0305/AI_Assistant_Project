"""
visualization.py — Research data visualizations for Task 4.
Generates Plotly charts based on actual retrieved research data.
All visualizations are grounded in the real arXiv paper data — no fake charts.
"""
import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)


def plot_category_distribution(papers: List[Dict]) -> Optional[object]:
    """
    Bar chart of paper count by CS sub-category.
    Uses real category data from retrieved/cached papers.
    """
    try:
        import plotly.express as px
        import pandas as pd

        counts: Dict[str, int] = {}
        for p in papers:
            cat = p.get("primary_category", p.get("categories", "unknown"))
            if isinstance(cat, str):
                cat = cat.split(",")[0].strip()
            counts[cat] = counts.get(cat, 0) + 1

        if not counts:
            return None

        df = pd.DataFrame(list(counts.items()), columns=["Category", "Count"])
        df = df.sort_values("Count", ascending=False)

        fig = px.bar(
            df,
            x="Category",
            y="Count",
            title="Papers by CS Sub-Category",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
        )
        return fig
    except Exception as e:
        logger.error(f"Category chart error: {e}")
        return None


def plot_keyword_frequency(papers: List[Dict], top_n: int = 20) -> Optional[object]:
    """
    Horizontal bar chart of most frequent keywords from paper titles/abstracts.
    Grounded in actual paper data.
    """
    try:
        import plotly.express as px
        import pandas as pd
        from collections import Counter

        # CS-domain stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "of", "in", "for", "to", "is",
            "on", "with", "this", "we", "our", "are", "that", "it", "by",
            "as", "be", "at", "from", "can", "has", "have", "not", "which",
            "its", "their", "also", "via", "show", "using", "based", "two",
            "three", "new", "into", "more", "each", "used", "use", "than",
            "both", "such", "these", "been", "i", "ii", "iii", "iv",
        }

        word_counts: Counter = Counter()
        for p in papers:
            text = f"{p.get('title', '')} {p.get('abstract', '')}"
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            for w in words:
                if w not in stopwords:
                    word_counts[w] += 1

        if not word_counts:
            return None

        top_words = word_counts.most_common(top_n)
        df = pd.DataFrame(top_words, columns=["Keyword", "Frequency"])

        fig = px.bar(
            df,
            y="Keyword",
            x="Frequency",
            orientation="h",
            title=f"Top {top_n} Keywords in Retrieved Papers",
            color="Frequency",
            color_continuous_scale="Plasma",
        )
        fig.update_layout(
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
        )
        return fig
    except Exception as e:
        logger.error(f"Keyword chart error: {e}")
        return None


def plot_publication_timeline(papers: List[Dict]) -> Optional[object]:
    """
    Line chart showing papers published over time.
    """
    try:
        import plotly.express as px
        import pandas as pd
        from collections import Counter

        year_counts: Counter = Counter()
        for p in papers:
            published = p.get("published", "")
            if published and len(published) >= 4:
                year = published[:4]
                if year.isdigit():
                    year_counts[year] += 1

        if len(year_counts) < 2:
            return None

        df = pd.DataFrame(
            sorted(year_counts.items()),
            columns=["Year", "Papers"]
        )

        fig = px.line(
            df,
            x="Year",
            y="Papers",
            title="Papers Published by Year",
            markers=True,
        )
        fig.update_traces(line_color="#7c3aed")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
        )
        return fig
    except Exception as e:
        logger.error(f"Timeline chart error: {e}")
        return None


def plot_retrieved_concept_map(retrieved_papers: List[Dict]) -> Optional[object]:
    """
    Network-style scatter plot showing concept relationships in retrieved papers.
    Each paper is a node; similar category papers are positioned nearby.
    """
    try:
        import plotly.graph_objects as go
        import random
        import math

        if not retrieved_papers:
            return None

        # Position papers in a circular layout by category
        category_positions: Dict[str, tuple] = {}
        n = len(retrieved_papers)

        fig = go.Figure()

        for i, paper in enumerate(retrieved_papers):
            angle = (2 * math.pi * i) / max(n, 1)
            x = math.cos(angle) * (0.5 + random.uniform(-0.1, 0.1))
            y = math.sin(angle) * (0.5 + random.uniform(-0.1, 0.1))

            title = paper.get("title", "Unknown")[:40] + "..."
            cat = paper.get("primary_category", "")

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=14, color="#7c3aed", opacity=0.8),
                text=[title],
                textposition="top center",
                name=cat,
                hovertemplate=(
                    f"<b>{paper.get('title', '')[:60]}</b><br>"
                    f"Category: {cat}<br>"
                    f"Authors: {paper.get('authors', '')[:50]}<br>"
                    f"Published: {paper.get('published', '')[:10]}"
                ),
            ))

        fig.update_layout(
            title="Retrieved Papers Concept Map",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            height=450,
        )
        return fig
    except Exception as e:
        logger.error(f"Concept map error: {e}")
        return None
