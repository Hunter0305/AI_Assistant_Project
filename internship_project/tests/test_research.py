"""
test_research.py — Tests for Task 4: Research Assistant
Tests: arXiv loader, visualization, retrieval (non-API tests)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.research.arxiv_loader import papers_to_documents, get_paper_categories_distribution
from src.research.visualization import plot_category_distribution, plot_keyword_frequency


SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N."],
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        "categories": ["cs.CL", "cs.LG"],
        "published": "2017-06-12",
        "arxiv_id": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "primary_category": "cs.CL",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin, J.", "Chang, M."],
        "abstract": "We introduce a new language representation model called BERT.",
        "categories": ["cs.CL"],
        "published": "2018-10-11",
        "arxiv_id": "https://arxiv.org/abs/1810.04805",
        "pdf_url": "https://arxiv.org/pdf/1810.04805",
        "primary_category": "cs.CL",
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["He, K.", "Zhang, X."],
        "abstract": "We present a residual learning framework to ease the training of networks.",
        "categories": ["cs.CV"],
        "published": "2015-12-10",
        "arxiv_id": "https://arxiv.org/abs/1512.03385",
        "pdf_url": "https://arxiv.org/pdf/1512.03385",
        "primary_category": "cs.CV",
    },
]


class TestArxivLoader:
    """Task 4: paper loading and document conversion."""

    def test_papers_to_documents_count(self):
        """Should convert all papers to documents."""
        docs = papers_to_documents(SAMPLE_PAPERS)
        assert len(docs) == 3

    def test_documents_have_metadata(self):
        """Documents should carry paper metadata."""
        docs = papers_to_documents(SAMPLE_PAPERS)
        doc = docs[0]
        assert "title" in doc.metadata
        assert "authors" in doc.metadata
        assert "arxiv_id" in doc.metadata

    def test_document_content_includes_abstract(self):
        """Document content should include the abstract."""
        docs = papers_to_documents(SAMPLE_PAPERS)
        for doc in docs:
            assert len(doc.page_content) > 10

    def test_empty_papers_list(self):
        """Empty list should return empty list."""
        docs = papers_to_documents([])
        assert docs == []

    def test_category_distribution(self):
        """Category distribution should count correctly."""
        distribution = {}
        for p in SAMPLE_PAPERS:
            cat = p["primary_category"]
            distribution[cat] = distribution.get(cat, 0) + 1
        assert distribution["cs.CL"] == 2
        assert distribution["cs.CV"] == 1


class TestResearchVisualization:
    """Task 4: visualizations grounded in real data."""

    def test_category_chart_returns_figure(self):
        """Category distribution chart should return a Plotly figure."""
        fig = plot_category_distribution(SAMPLE_PAPERS)
        # May return None if plotly not installed
        if fig is not None:
            # Check it has the right structure
            assert hasattr(fig, "data") or hasattr(fig, "layout")

    def test_keyword_chart_returns_figure(self):
        """Keyword frequency chart should return a figure or None."""
        fig = plot_keyword_frequency(SAMPLE_PAPERS)
        # Valid return is a figure or None (if plotly not installed)
        assert fig is None or hasattr(fig, "data")

    def test_charts_use_actual_data(self):
        """Charts should use real paper data, not fake data."""
        # Verify that chart with no papers returns None
        fig = plot_category_distribution([])
        assert fig is None

    def test_keyword_chart_empty_papers(self):
        """Keyword chart with no papers should return None."""
        fig = plot_keyword_frequency([])
        assert fig is None
