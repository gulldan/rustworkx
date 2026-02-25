"""Implementation aggregator for plotting helpers."""

from plotting_chart import create_comparison_chart
from plotting_gcr import plot_gcr_pcp_vs_jaccard_threshold
from plotting_shared import (
    ALGORITHMS_METADATA,
    BenchmarkResultItem,
    BenchmarkResultsList,
    EXACT_MATCH_MPL_HEADER_LABELS,
    FORMATTER_FUNCTIONS,
    METRIC_KEY_TO_PROPERTIES,
    MetricProperty,
    TABLE_COLUMN_KEYS,
    TABLE_COLUMN_NAMES,
    TABLE_COLUMN_NAMES_MPL,
    _coerce_float_metric_value,
    _format_metric_cell_value,
    _get_algo_run_elapsed_and_presence,
    get_valid_benchmark_results,
    style_exact_match_markdown_cell,
)
from plotting_tables import (
    generate_results_table,
    generate_results_table_matplotlib,
    get_color_for_cell_mpl,
)

__all__ = [
    "BenchmarkResultItem",
    "BenchmarkResultsList",
    "MetricProperty",
    "TABLE_COLUMN_KEYS",
    "TABLE_COLUMN_NAMES",
    "METRIC_KEY_TO_PROPERTIES",
    "EXACT_MATCH_MPL_HEADER_LABELS",
    "TABLE_COLUMN_NAMES_MPL",
    "ALGORITHMS_METADATA",
    "FORMATTER_FUNCTIONS",
    "_get_algo_run_elapsed_and_presence",
    "_coerce_float_metric_value",
    "_format_metric_cell_value",
    "style_exact_match_markdown_cell",
    "get_valid_benchmark_results",
    "create_comparison_chart",
    "generate_results_table",
    "get_color_for_cell_mpl",
    "generate_results_table_matplotlib",
    "plot_gcr_pcp_vs_jaccard_threshold",
]
