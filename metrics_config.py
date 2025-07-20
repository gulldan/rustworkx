# from benchmark_utils import format_memory, format_time # This line caused circular import

# Constants that should align with benchmark_community.py
# LARGE_GRAPH_THRESHOLD = 94000 # Not needed for metrics definition
# RESOLUTIONS_TO_TEST = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0] # Not needed directly for ORDERED_METRIC_PROPERTIES structure

JACCARD_THRESHOLDS_TO_TEST = [0.25, 0.5, 0.75]

# --- Ordered Metric Properties: Single source of truth for table columns and plot metrics ---
# 'key': internal key for data lookup (suffix for algorithm prefix)
# 'name': display name for table headers and plot titles
# 'type': 'info', 'perf', 'structure', 'internal', 'external', 'summary'
# 'higher_is_better': True, False, or None (if not applicable for coloring/scoring)
# 'format_func': optional custom formatting function for display
# The actual functions for format_func (like format_time, format_memory) are expected
# to be available in the scope where ORDERED_METRIC_PROPERTIES is processed for display (e.g., in plotting.py)
ORDERED_METRIC_PROPERTIES = [
    {"key": "dataset", "name": "Dataset", "type": "info", "higher_is_better": None},
    {"key": "nodes", "name": "Nodes", "type": "info", "higher_is_better": None},
    {"key": "edges", "name": "Edges", "type": "info", "higher_is_better": None},
    {"key": "num_gt_clusters", "name": "GT Clusters", "type": "info", "higher_is_better": None},
    {"key": "num_nodes_in_gt", "name": "Nodes in GT", "type": "info", "higher_is_better": None},
    {"key": "algorithm_name", "name": "Algorithm", "type": "info", "higher_is_better": None},
    {
        "key": "_elapsed",
        "name": "Time",
        "type": "perf",
        "higher_is_better": False,
        "format_func": "format_time",
    },  # Store name of func
    {
        "key": "_memory",
        "name": "Memory",
        "type": "perf",
        "higher_is_better": False,
        "format_func": "format_memory",
    },  # Store name of func
    {"key": "_num_comms", "name": "# Pred Comms", "type": "structure", "higher_is_better": None},
    {
        "key": "_num_singleton_comms",
        "name": "% Small Comms (<5)",
        "type": "structure",
        "higher_is_better": False,
    },
    {"key": "_common_nodes", "name": "# Common Nodes", "type": "structure", "higher_is_better": True},
    {
        "key": "_unclustered_pct",
        "name": "Unclustered Pct (%)",
        "type": "structure",
        "higher_is_better": False,
    },
    {"key": "_modularity", "name": "Modularity", "type": "internal", "higher_is_better": True},
    {"key": "_conductance", "name": "Conductance", "type": "internal", "higher_is_better": False},
    {"key": "_internal_density", "name": "Int. Density", "type": "internal", "higher_is_better": True},
    {
        "key": "_avg_internal_degree",
        "name": "Avg. Int. Deg.",
        "type": "internal",
        "higher_is_better": True,
    },
    {"key": "_tpr", "name": "TPR", "type": "internal", "higher_is_better": True},
    {"key": "_cut_ratio", "name": "Cut Ratio", "type": "internal", "higher_is_better": False},
    {"key": "_surprise", "name": "Surprise", "type": "internal", "higher_is_better": True},
    {"key": "_significance", "name": "Significance", "type": "internal", "higher_is_better": True},
    {"key": "_ari", "name": "ARI", "type": "external", "higher_is_better": True},
    {"key": "_nmi", "name": "NMI", "type": "external", "higher_is_better": True},
    {"key": "_v_measure", "name": "V-Measure", "type": "external", "higher_is_better": True},
    {"key": "_fmi", "name": "FMI", "type": "external", "higher_is_better": True},
    {"key": "_purity", "name": "Purity", "type": "external", "higher_is_better": True},
    {"key": "_vi", "name": "VI", "type": "external", "higher_is_better": False},
    {"key": "_pw_precision", "name": "PW Prec.", "type": "external", "higher_is_better": True},
    {"key": "_pw_recall", "name": "PW Rec.", "type": "external", "higher_is_better": True},
    {"key": "_pw_f1", "name": "PW F1", "type": "external", "higher_is_better": True},
]

# Dynamically add Jaccard threshold metrics
for jt_val in JACCARD_THRESHOLDS_TO_TEST:
    jt_str_key = str(jt_val).replace(".", "p")
    ORDERED_METRIC_PROPERTIES.append(
        {
            "key": f"_gcr_jt{jt_str_key}",
            "name": f"GT Rec. ({jt_val})",
            "type": "external",
            "higher_is_better": True,
        }
    )
    ORDERED_METRIC_PROPERTIES.append(
        {
            "key": f"_pcp_jt{jt_str_key}",
            "name": f"Pred Prec. ({jt_val})",
            "type": "external",
            "higher_is_better": True,
        }
    )

# Add Overall Score to the list of metrics
ORDERED_METRIC_PROPERTIES.append(
    {"key": "_overall_score", "name": "Overall Score", "type": "summary", "higher_is_better": True}
)
