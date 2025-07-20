# benchmark_config_data.py

from collections.abc import Callable
from typing import Any

from dataset_loaders import (
    load_citeseer,
    load_cora,
    load_davis_women,
    load_dolphins,
    load_email_eu_core,
    load_facebook,
    load_florentine_families,
    load_football,
    load_graph_edges_no_gt_polars,
    load_karate_club,
    load_les_miserables,
    load_polblogs,
    load_political_books,
    load_wiki_news_sim,
)

# List of resolution parameters to test for tunable algorithms
# RESOLUTIONS_TO_TEST = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
RESOLUTIONS_TO_TEST = [1.0]

LARGE_GRAPH_THRESHOLD = 94000
# Skip NetworkX algorithms for datasets with more than this many edges (too slow)
LARGE_EDGES_THRESHOLD = 1_000_000
# --- Algorithm Color Definitions ---
NX_LOUVAIN_COLOR = "#1f77b4"  # Muted Blue
RX_LOUVAIN_COLOR = "#ff7f0e"  # Safety Orange
RX_LEIDEN_COLOR = "#2ca02c"  # Cooked Asparagus Green
RX_LPA_U_COLOR = "#d62728"  # Brick Red
RX_LPA_W_COLOR = "#9467bd"  # Muted Purple
CDLIB_LEIDEN_COLOR = "#8c564b"  # Chestnut Brown
NX_LPA_COLOR = "#e377c2"  # Pink
GRID_COLOR = "#E5E5E5"  # Plotting grid color

# --- Centralized Algorithm Configuration Structure ---
# This list defines all algorithms, their parameterizations, runner function *names*,
# and display properties.
# Runner function names will be resolved to actual functions in benchmark_community.py
ALGORITHMS_CONFIG_STRUCTURE = []

# Setup for tunable algorithms
_TUNABLE_ALGOS_SETUP = [
    {
        "base_prefix": "nx_louvain",
        "base_name": "NX Louvain",
        "color": NX_LOUVAIN_COLOR,
        "runner_name": "run_nx_algorithm",
        "is_rx": False,
    },
    {
        "base_prefix": "rx_louvain",
        "base_name": "RX Louvain",
        "color": RX_LOUVAIN_COLOR,
        "runner_name": "run_rx_algorithm",
        "is_rx": True,
    },
    {
        "base_prefix": "rx_leiden",
        "base_name": "RX Leiden",
        "color": RX_LEIDEN_COLOR,
        "runner_name": "run_rx_leiden_algorithm",
        "is_rx": True,
    },
]
for algo_setup in _TUNABLE_ALGOS_SETUP:
    for res_val in RESOLUTIONS_TO_TEST:
        res_str_key = str(res_val).replace(".", "p")
        ALGORITHMS_CONFIG_STRUCTURE.append(
            {
                "prefix": f"{algo_setup['base_prefix']}_res{res_str_key}",
                "name": f"{algo_setup['base_name']} (res {res_val})",
                "color": algo_setup["color"],
                "base_prefix": algo_setup["base_prefix"],
                "runner_name": algo_setup["runner_name"],
                "is_rx": algo_setup["is_rx"],
                "run_args": {"resolution": res_val},
            }
        )

# Setup for non-tunable algorithms
_NON_TUNABLE_ALGOS_SETUP = [
    {
        "prefix": "cdlib_leiden",
        "name": "cdlib Leiden",
        "color": CDLIB_LEIDEN_COLOR,
        "runner_name": "run_cdlib_leiden",
        "is_rx": False,
        "run_args": {},
    },
    {
        "prefix": "rx_lpa_unweighted",
        "name": "RX LPA (U)",
        "color": RX_LPA_U_COLOR,
        "runner_name": "run_rx_lpa_algorithm",
        "is_rx": True,
        "run_args": {"weighted": False, "seed": 42},
    },
    {
        "prefix": "rx_lpa_weighted",
        "name": "RX LPA (W)",
        "color": RX_LPA_W_COLOR,
        "runner_name": "run_rx_lpa_algorithm",
        "is_rx": True,
        "run_args": {"weighted": True, "seed": 42},
    },
    {
        "prefix": "nx_lpa",
        "name": "NX LPA",
        "color": NX_LPA_COLOR,
        "runner_name": "run_nx_lpa_algorithm",
        "is_rx": False,
        "run_args": {"seed": 42},
    },
]
for algo_setup in _NON_TUNABLE_ALGOS_SETUP:
    ALGORITHMS_CONFIG_STRUCTURE.append(
        {
            "prefix": algo_setup["prefix"],
            "name": algo_setup["name"],
            "color": algo_setup["color"],
            "base_prefix": algo_setup["prefix"],
            "runner_name": algo_setup["runner_name"],
            "is_rx": algo_setup["is_rx"],
            "run_args": algo_setup["run_args"],
        }
    )

# Note: JACCARD_THRESHOLDS_TO_TEST and ORDERED_METRIC_PROPERTIES remain in metrics_config.py
# as they are specific to metrics and not general benchmark/algorithm structure.


# --- Dataset Configuration ---

# Dictionary of datasets to run the benchmark on.
# The key is the dataset name (for display) and the value is the loading function.
# NOTE: The benchmark script is designed to work with loaders that return a dictionary.
# Loaders returning tuples will require modification of the benchmark script to work.
DATASETS: dict[str, Callable[..., Any]] = {
    # Dict-based loaders (compatible with current benchmark script)
    "Wiki News Sim": load_wiki_news_sim,

    # Temporarily disable other datasets for focused testing
    # "Graph Edges GT Clusters": load_graph_edges_gt_clusters,
    # "Graph Edges LLM Clusters": load_graph_edges_llm_clusters,
    "Graph Edges No GT": load_graph_edges_no_gt_polars,

    # Simple loaders (currently incompatible, require benchmark script modification)
    "Karate Club": load_karate_club,
    "Davis Southern Women": load_davis_women,
    "Florentine Families": load_florentine_families,
    "Les Misérables": load_les_miserables,
    "American College Football": load_football,
    "Political Books": load_political_books,
    "Dolphins": load_dolphins,
    "Political Blogs": load_polblogs,
    "Cora": load_cora,
    "Facebook": load_facebook,
    "Citeseer": load_citeseer,
    "Email EU Core": load_email_eu_core,
    # "Wiki News": load_graph_edges_no_gt_polars,
    # "Graph Edges (CSV)": load_graph_edges_csv,
    # "Graph Edges (Parquet)": load_graph_edges_parquet,
    # "Graph Edges (9M Parquet)": load_graph_edges_9m_parquet,
    # "LFR (n=250, mu=0.1)": lambda: load_lfr(n=250, mu=0.1),
    # "LFR (n=1000, mu=0.3)": lambda: load_lfr(n=1000, mu=0.3),
    # "Large Synthetic (SBM)": load_large_synthetic,
    # "Orkut": load_orkut,
    # "LiveJournal": load_livejournal,
}

# Dictionary to specify datasets to skip.
# Key is the dataset name (must match a key in DATASETS).
# Value is True to skip.
SKIPPED_DATASETS: dict[str, bool] = {
    # "Orkut": True, # Example: Skip Orkut because it is very large
    # "LiveJournal": True,
}
