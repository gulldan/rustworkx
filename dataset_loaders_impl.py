"""Implementation aggregator for dataset loader functions."""

from dataset_loaders_builtin import (
    load_citeseer,
    load_cora,
    load_davis_women,
    load_dolphins,
    load_email_eu_core,
    load_facebook,
    load_florentine_families,
    load_football,
    load_karate_club,
    load_les_miserables,
    load_polblogs,
    load_political_books,
)
from dataset_loaders_parquet import (
    load_graph_edges_9m_parquet,
    load_graph_edges_csv,
    load_graph_edges_gt_clusters,
    load_graph_edges_llm_clusters,
    load_graph_edges_no_gt_polars,
    load_graph_edges_parquet,
    load_wiki_news_edges,
)
from dataset_loaders_shared import (
    DictLoaderReturn,
    SimpleLoaderReturn,
    _add_weighted_edges_from_df,
    _ensure_default_edge_weight,
    _load_gml_with_ground_truth,
)
from dataset_loaders_snap import load_livejournal, load_orkut, load_snap_text_dataset
from dataset_loaders_synthetic import load_large_synthetic, load_lfr

__all__ = [
    "SimpleLoaderReturn",
    "DictLoaderReturn",
    "_ensure_default_edge_weight",
    "_load_gml_with_ground_truth",
    "_add_weighted_edges_from_df",
    "load_karate_club",
    "load_davis_women",
    "load_florentine_families",
    "load_les_miserables",
    "load_football",
    "load_political_books",
    "load_dolphins",
    "load_polblogs",
    "load_cora",
    "load_facebook",
    "load_citeseer",
    "load_email_eu_core",
    "load_graph_edges_csv",
    "load_graph_edges_parquet",
    "load_graph_edges_9m_parquet",
    "load_lfr",
    "load_large_synthetic",
    "load_snap_text_dataset",
    "load_orkut",
    "load_livejournal",
    "load_graph_edges_gt_clusters",
    "load_graph_edges_llm_clusters",
    "load_wiki_news_edges",
    "load_graph_edges_no_gt_polars",
]
