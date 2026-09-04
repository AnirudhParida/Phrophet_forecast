"""
pipeline/visualization.py
==========================
Visualization utilities for the NeuralProphet infrastructure forecasting
pipeline.

Generates publication-quality time-series forecast charts saved as PNG files
under ``outputs/forecasts/``.

Chart anatomy
-------------
For each host × metric combination the chart shows:
  - Historical actuals (full 366-day period, steel-blue line)
  - 7-day forecast values (yhat1 through yhat7, orange dashed line)
  - 90 % prediction interval (shaded band between 5th and 95th quantile)
  - Vertical dashed line marking the train/forecast boundary
  - Metric statistics annotated in a text box (mean, std, min, max)

Styling
-------
Uses a dark background (``"dark_background"`` matplotlib style) with a
carefully chosen colour palette so charts look professional alongside the
pipeline's log output.

Output
------
All figures are saved to ``FORECASTS_DIR/{host_alias}_{metric}_forecast.png``
at 150 DPI.  A summary grid image (3 metrics in one figure per host) is also
saved as ``{host_alias}_overview.png``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import FORECASTS_DIR, METRIC_LABELS, METRICS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (chosen for dark-background legibility)
# ---------------------------------------------------------------------------
_PALETTE = {
    "actual": "#5B9BD5",       # steel blue
    "forecast": "#FF8C00",     # dark orange
    "interval": "#FF8C00",     # same hue as forecast, low alpha
    "boundary": "#E0E0E0",     # light grey
    "grid": "#444444",
    "text_box": "#1E1E2E",
    "text_fg": "#CDD6F4",
}

_DPI = 150
_FIGSIZE_SINGLE = (14, 5)
_FIGSIZE_OVERVIEW = (16, 12)


def get_y_formatter(metric: str):
    def pct_formatter(x, _): return f"{x:.1f}%"
    def bytes_formatter(x, _):
        if x >= 1024**3: return f"{x / 1024**3:.1f} GB"
        if x >= 1024**2: return f"{x / 1024**2:.1f} MB"
        if x >= 1024: return f"{x / 1024:.1f} KB"
        return f"{x:.0f} B"
    def ops_formatter(x, _): return f"{x:.1f}"

    if metric.endswith("_pct"):
        return plt.FuncFormatter(pct_formatter)
    elif "bytes" in metric:
        return plt.FuncFormatter(bytes_formatter)
    else:
        return plt.FuncFormatter(ops_formatter)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_forecast(
    host_alias: str,
    metric: str,
    actuals_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    n_lags: int,
    save: bool = True,
) -> Path:
    """
    Generate a single forecast chart for one host × metric combination.

    Parameters
    ----------
    host_alias  : Short host alias (e.g. ``"HYDUPINTAPP16"``).
    metric      : Metric column name (e.g. ``"cpu_pct"``).
    actuals_df  : Full historical DataFrame with columns ``ds`` and ``{metric}``.
    forecast_df : NeuralProphet prediction DataFrame.  Must contain ``ds``,
                  ``yhat1``, and optionally quantile columns.
    n_lags      : n_lags used in training (shown in subtitle).
    save        : If True, save the figure to ``FORECASTS_DIR``.

    Returns
    -------
    Path to the saved PNG (or empty Path if ``save=False``).
    """
    label = METRIC_LABELS.get(metric, metric)
    title = f"{host_alias}  —  {label}  (n_lags={n_lags})"

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
        fig.patch.set_facecolor("#13131A")
        ax.set_facecolor("#1C1C28")

        # --- Historical actuals ---
        ax.plot(
            actuals_df["ds"],
            actuals_df[metric],
            color=_PALETTE["actual"],
            linewidth=1.4,
            label="Historical actuals",
            alpha=0.9,
        )

        # Extract clean point predictions and quantiles (future window only)
        forecast_rows = _extract_forecast_rows(forecast_df, metric)
        if forecast_rows is not None and len(forecast_rows) > 0:
            ax.plot(
                forecast_rows["ds"],
                forecast_rows["yhat"],
                color=_PALETTE["forecast"],
                linewidth=2.0,
                linestyle="--",
                marker="o",
                markersize=4,
                label=f"{len(forecast_rows)}-day forecast",
                zorder=5,
            )

            # Confidence band (if quantile columns exist)
            if "yhat_lower" in forecast_rows.columns and "yhat_upper" in forecast_rows.columns:
                ax.fill_between(
                    forecast_rows["ds"],
                    forecast_rows["yhat_lower"],
                    forecast_rows["yhat_upper"],
                    color=_PALETTE["interval"],
                    alpha=0.18,
                    label="90% confidence interval",
                )

            # --- Boundary line ---
            boundary_date = actuals_df["ds"].max()
            ax.axvline(
                x=boundary_date,
                color=_PALETTE["boundary"],
                linewidth=1.0,
                linestyle=":",
                alpha=0.6,
                label="Forecast start",
            )

        # --- Formatting ---
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)
        ax.yaxis.set_major_formatter(get_y_formatter(metric))
        ax.set_ylabel(label, color=_PALETTE["text_fg"], fontsize=10)
        ax.set_title(title, color=_PALETTE["text_fg"], fontsize=12, pad=12)
        ax.tick_params(colors=_PALETTE["text_fg"])
        ax.grid(True, color=_PALETTE["grid"], linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(_PALETTE["grid"])

        # --- Stats annotation box ---
        _add_stats_box(ax, actuals_df[metric], _PALETTE, metric=metric)

        legend = ax.legend(
            loc="upper left",
            fontsize=8,
            framealpha=0.3,
            edgecolor=_PALETTE["grid"],
            facecolor=_PALETTE["text_box"],
            labelcolor=_PALETTE["text_fg"],
        )

        plt.tight_layout()

        if save:
            FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = FORECASTS_DIR / f"{host_alias}_{metric}_lags{n_lags}_forecast.png"
            fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved forecast chart → %s", out_path)
            plt.close(fig)
            return out_path

        plt.show()
        plt.close(fig)
        return Path()


def plot_overview(
    host_alias: str,
    actuals_df: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    n_lags: int,
    save: bool = True,
) -> Path:
    """
    Generate a 3-row overview chart showing all metrics for one host.

    Parameters
    ----------
    host_alias : Short host alias.
    actuals_df : Full historical DataFrame (ds, cpu_pct, memory_pct, disk_pct).
    forecasts  : Dict mapping metric name → NeuralProphet prediction DataFrame.
    n_lags     : n_lags used in training.
    save       : If True, save to ``FORECASTS_DIR``.

    Returns
    -------
    Path to the saved PNG.
    """
    n_metrics = len(METRICS)
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(n_metrics, 1, figsize=_FIGSIZE_OVERVIEW, sharex=True)
        fig.patch.set_facecolor("#13131A")
        fig.suptitle(
            f"{host_alias}  —  Infrastructure Metrics Forecast  (n_lags={n_lags})",
            fontsize=14,
            color=_PALETTE["text_fg"],
            y=0.99,
        )

        for ax, metric in zip(axes, METRICS):
            ax.set_facecolor("#1C1C28")
            label = METRIC_LABELS.get(metric, metric)

            # Historical
            ax.plot(
                actuals_df["ds"],
                actuals_df[metric],
                color=_PALETTE["actual"],
                linewidth=1.3,
                alpha=0.9,
                label="Actuals",
            )

            # Forecast
            if metric in forecasts:
                forecast_rows = _extract_forecast_rows(forecasts[metric], metric)
                if forecast_rows is not None and len(forecast_rows) > 0:
                    ax.plot(
                        forecast_rows["ds"],
                        forecast_rows["yhat"],
                        color=_PALETTE["forecast"],
                        linewidth=2.0,
                        linestyle="--",
                        marker="o",
                        markersize=3,
                        label="Forecast",
                        zorder=5,
                    )
                    if "yhat_lower" in forecast_rows.columns and "yhat_upper" in forecast_rows.columns:
                        ax.fill_between(
                            forecast_rows["ds"],
                            forecast_rows["yhat_lower"],
                            forecast_rows["yhat_upper"],
                            color=_PALETTE["interval"],
                            alpha=0.15,
                            label="90% CI",
                        )
                    ax.axvline(
                        x=actuals_df["ds"].max(),
                        color=_PALETTE["boundary"],
                        linewidth=0.8,
                        linestyle=":",
                        alpha=0.5,
                    )

            ax.set_ylabel(label, color=_PALETTE["text_fg"], fontsize=9)
            ax.yaxis.set_major_formatter(get_y_formatter(metric))
            ax.tick_params(colors=_PALETTE["text_fg"], labelsize=8)
            ax.grid(True, color=_PALETTE["grid"], linewidth=0.4, alpha=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor(_PALETTE["grid"])
            _add_stats_box(ax, actuals_df[metric], _PALETTE, fontsize=7, metric=metric)
            ax.legend(loc="upper left", fontsize=7, framealpha=0.2,
                      facecolor=_PALETTE["text_box"], labelcolor=_PALETTE["text_fg"])

        # Shared x-axis formatting
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(axes[-1].get_xticklabels(), rotation=15, ha="right", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save:
            FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = FORECASTS_DIR / f"{host_alias}_overview_lags{n_lags}.png"
            fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved overview chart → %s", out_path)
            plt.close(fig)
            return out_path

        plt.show()
        plt.close(fig)
        return Path()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def extract_forecast_rows(
    df: pd.DataFrame, metric: str = None, n_steps: int = 30
) -> pd.DataFrame | None:
    """
    Extract clean future forecast rows (ds, yhat, yhat_lower, yhat_upper)
    from a NeuralProphet prediction DataFrame.
    """
    if df is None or df.empty:
        return None

    # Find yhat columns
    yhat_cols = sorted(
        [c for c in df.columns if c.startswith("yhat") and c[4:].isdigit()],
        key=lambda c: int(c[4:]),
    )
    if not yhat_cols:
        return None

    # Last historical row (where y is known)
    if "y" in df.columns:
        last_hist_idx = df["y"].last_valid_index()
    else:
        last_hist_idx = df.index[-len(yhat_cols) - 1]

    if last_hist_idx is None:
        return None

    last_row = df.loc[last_hist_idx]
    last_ds = pd.to_datetime(last_row["ds"])
    n_steps = len(yhat_cols)

    future_dates = [last_ds + pd.Timedelta(days=i + 1) for i in range(n_steps)]
    yhat_values = [last_row[col] for col in yhat_cols]

    result = pd.DataFrame({"ds": future_dates, "yhat": yhat_values})

    # Quantiles
    lower_cols = sorted(
        [c for c in df.columns if "5.0%" in c and c.startswith("yhat")],
        key=lambda c: int(c.split(" ")[0][4:]),
    )
    upper_cols = sorted(
        [c for c in df.columns if "95.0%" in c and c.startswith("yhat")],
        key=lambda c: int(c.split(" ")[0][4:]),
    )

    if lower_cols and upper_cols:
        result["yhat_lower"] = [last_row[col] for col in lower_cols[:n_steps]]
        result["yhat_upper"] = [last_row[col] for col in upper_cols[:n_steps]]

    # Clip values: percentages to [0.0, 100.0], others to [0.0, None]
    upper_bound = 100.0 if metric is None or metric.endswith("_pct") else None

    result["yhat"] = result["yhat"].clip(0.0, upper_bound)
    if "yhat_lower" in result.columns and "yhat_upper" in result.columns:
        raw_lower = result["yhat_lower"].clip(0.0, upper_bound)
        raw_upper = result["yhat_upper"].clip(0.0, upper_bound)
        fixed_lower = raw_lower.combine(raw_upper, min)
        fixed_upper = raw_lower.combine(raw_upper, max)
        result["yhat_lower"] = np.minimum(fixed_lower.values, result["yhat"].values)
        result["yhat_upper"] = np.maximum(fixed_upper.values, result["yhat"].values)
    elif "yhat_lower" in result.columns:
        result["yhat_lower"] = result["yhat_lower"].clip(0.0, upper_bound)
    elif "yhat_upper" in result.columns:
        result["yhat_upper"] = result["yhat_upper"].clip(0.0, upper_bound)

    return result

_extract_forecast_rows = extract_forecast_rows




def _add_stats_box(
    ax: plt.Axes,
    series: pd.Series,
    palette: dict,
    fontsize: int = 8,
    metric: str = "pct",
) -> None:
    """
    Annotate the top-right corner of an axes with descriptive statistics.
    """
    unit = "%" if metric.endswith("_pct") else ""
    if "bytes" in metric:
        unit = " B"
    
    stats_text = (
        f"μ={series.mean():.2f}{unit}  "
        f"σ={series.std():.2f}{unit}\n"
        f"min={series.min():.2f}{unit}  "
        f"max={series.max():.2f}{unit}"
    )
    ax.text(
        0.99, 0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        horizontalalignment="right",
        color=palette["text_fg"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=palette["text_box"],
            alpha=0.6,
            edgecolor=palette["grid"],
        ),
    )
