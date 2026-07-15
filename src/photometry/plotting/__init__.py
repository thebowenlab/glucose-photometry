from photometry.plotting.group_average import (
    build_matrix,
    compute_group_average,
    plot_group_average,
)
from photometry.plotting.comparison import (
    plot_dataset_comparison,
    plot_two_group_bar,
)
from photometry.plotting.session import plot_session_summary
from photometry.plotting.cohort import prepare_session_matrix, plot_multi_signal_summary
from photometry.plotting.correlation import plot_cross_correlation_summary
from photometry.plotting.metrics import plot_metrics_by_group

__all__ = [
    "build_matrix",
    "compute_group_average",
    "plot_group_average",
    "plot_dataset_comparison",
    "plot_two_group_bar",
    "plot_session_summary",
    "prepare_session_matrix",
    "plot_multi_signal_summary",
    "plot_cross_correlation_summary",
    "plot_metrics_by_group",
]
