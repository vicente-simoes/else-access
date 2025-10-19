
from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy.optimize import OptimizeWarning, curve_fit


REPO_ROOT = Path(__file__).resolve().parent


def usl_model(N: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return N / (1 + alpha * (N - 1) + beta * N * (N - 1))


@dataclass(frozen=True)
class PlotSpec:
    key: str
    column: str
    ylabel: str
    title: str
    legend_label: str
    fit: str | None = None
    transform: Callable[[pd.DataFrame, str], pd.DataFrame] | None = None
    filename_stub: str | None = None


def _identity_transform(df: pd.DataFrame, _: str) -> pd.DataFrame:
    return df


def _speedup_transform(df: pd.DataFrame, scenario_label: str) -> pd.DataFrame:
    baseline_clients = df["clients"].min()
    baseline_mask = df["clients"] == baseline_clients
    if baseline_mask.sum() == 0:
        raise ValueError(
            f"Cannot compute speedup for {scenario_label}: no data for clients={baseline_clients}."
        )
    baseline = df.loc[baseline_mask, "value"].iloc[0]
    if baseline <= 0:
        raise ValueError(
            f"Cannot compute speedup for {scenario_label}: baseline throughput must be positive."
        )
    df = df.copy()
    df["value"] = df["value"] / baseline
    return df


def _efficiency_transform(df: pd.DataFrame, scenario_label: str) -> pd.DataFrame:
    df = _speedup_transform(df, scenario_label)
    df["value"] = df["value"] / df["clients"]
    return df


PLOT_TYPES: Dict[str, PlotSpec] = {
    "usl_throughput": PlotSpec(
        key="usl_throughput",
        column="tps",
        ylabel="Throughput (TPS)",
        title="CitusDB Scalability Analysis with USL",
        legend_label="Measured throughput",
        fit="usl",
        transform=_identity_transform,
        filename_stub="throughput-usl",
    ),
    "throughput_raw": PlotSpec(
        key="throughput_raw",
        column="tps",
        ylabel="Throughput (TPS)",
        title="CitusDB Throughput",
        legend_label="Measured throughput",
        transform=_identity_transform,
        filename_stub="throughput",
    ),
    "speedup": PlotSpec(
        key="speedup",
        column="tps",
        ylabel="Speedup",
        title="CitusDB Speedup vs. Client Count",
        legend_label="Speedup",
        transform=_speedup_transform,
        filename_stub="speedup",
    ),
    "efficiency": PlotSpec(
        key="efficiency",
        column="tps",
        ylabel="Parallel Efficiency",
        title="CitusDB Parallel Efficiency",
        legend_label="Efficiency",
        transform=_efficiency_transform,
        filename_stub="efficiency",
    ),
    "latency_mean": PlotSpec(
        key="latency_mean",
        column="lat_mean_ms",
        ylabel="Average Latency (ms)",
        title="CitusDB Average Latency",
        legend_label="Measured latency",
        transform=_identity_transform,
        filename_stub="latency-mean",
    ),
    "latency_p95": PlotSpec(
        key="latency_p95",
        column="lat_p95_ms",
        ylabel="p95 Latency (ms)",
        title="CitusDB p95 Latency",
        legend_label="Measured latency",
        transform=_identity_transform,
        filename_stub="latency-p95",
    ),
    "latency_p99": PlotSpec(
        key="latency_p99",
        column="lat_p99_ms",
        ylabel="p99 Latency (ms)",
        title="CitusDB p99 Latency",
        legend_label="Measured latency",
        transform=_identity_transform,
        filename_stub="latency-p99",
    ),
}


PLOT_ALIASES = {
    "throughput": "usl_throughput",
    "usl": "usl_throughput",
}


DEFAULT_SCENARIO_CANDIDATES = [
    "worker_group",
    "duration",
    "workers",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Universal Scalability Law and related plots from benchmark results."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results.csv",
        help=(
            "Path to the input CSV file relative to the repository root or as an absolute "
            "path. Defaults to 'results.csv'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "plots",
        help=(
            "Directory where generated plots are written. The directory is created when "
            "it does not exist. Defaults to 'plots/'."
        ),
    )
    plot_choices = sorted(set(PLOT_TYPES) | set(PLOT_ALIASES)) + ["all"]
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=plot_choices,
        default=["usl_throughput"],
        help=(
            "List of plots to generate. Choose one or more values from %(choices)s or "
            "use 'all' to render every available metric."
        ),
    )
    parser.add_argument(
        "--scenario-columns",
        nargs="*",
        help=(
            "Columns that differentiate the series within each scale-specific plot. The "
            "values are combined with worker information to label individual lines. When "
            "omitted the script selects informative columns automatically (for example "
            "worker group and benchmark duration)."
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help=(
            "Restrict the input data before plotting. Repeat the option to apply "
            "multiple filters. Values may be comma-separated to allow several matches."
        ),
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Label that identifies the benchmark run. It is embedded in output file "
            "names. Defaults to the CSV stem or timestamp when available."
        ),
    )
    parser.add_argument(
        "--title",
        help=(
            "Override the base plot title. Scenario details are appended automatically "
            "when more than one chart is generated."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def sanitize_for_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "plot"


def infer_run_identifier(df: pd.DataFrame, csv_path: Path) -> str:
    if "timestamp_utc" in df.columns:
        timestamps = [str(ts) for ts in df["timestamp_utc"].dropna().unique()]
        if timestamps:
            return sanitize_for_filename(sorted(timestamps)[0])
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_?\d{2}_?\d{2})", csv_path.stem)
    if match:
        return sanitize_for_filename(match.group(1))
    return sanitize_for_filename(csv_path.stem)


def normalize_plots(selections: Sequence[str]) -> List[str]:
    if "all" in selections:
        return list(PLOT_TYPES.keys())
    normalized: List[str] = []
    for item in selections:
        key = PLOT_ALIASES.get(item, item)
        if key not in PLOT_TYPES:
            raise KeyError(f"Unknown plot type '{item}'.")
        normalized.append(key)
    return normalized


def infer_scenario_columns(
    df: pd.DataFrame, requested: Sequence[str] | None
) -> List[str]:
    if requested is not None and len(requested) == 0:
        return []
    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            available = ", ".join(sorted(df.columns))
            raise ValueError(
                "Unknown scenario columns: {missing}. Available columns: {available}.".format(
                    missing=", ".join(missing), available=available
                )
            )
        columns = list(dict.fromkeys(requested))
    else:
        columns = []
        for candidate in DEFAULT_SCENARIO_CANDIDATES:
            if candidate in df.columns and df[candidate].nunique(dropna=False) > 1:
                columns.append(candidate)

    return [column for column in columns if column not in {"scale", "workers"}]


def parse_filter_value(raw: str, series: pd.Series) -> object:
    raw = raw.strip()
    dtype = series.dtype
    if ptypes.is_integer_dtype(dtype):
        return int(raw)
    if ptypes.is_float_dtype(dtype):
        return float(raw)
    if ptypes.is_bool_dtype(dtype):
        lowered = raw.lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            return True
        if lowered in {"0", "false", "f", "no", "n"}:
            return False
        raise ValueError(f"Cannot parse boolean value '{raw}'.")
    return raw


def apply_filters(df: pd.DataFrame, filters: Sequence[str]) -> pd.DataFrame:
    filtered = df
    for item in filters:
        if "=" not in item:
            raise ValueError(
                f"Invalid filter '{item}'. Expected the format COLUMN=VALUE[,VALUE2]."
            )
        column, values = item.split("=", 1)
        column = column.strip()
        if column not in filtered.columns:
            available = ", ".join(sorted(filtered.columns))
            raise ValueError(
                f"Cannot filter by '{column}'. Available columns: {available}."
            )
        series = filtered[column]
        allowed = [
            parse_filter_value(value, series) for value in values.split(",") if value.strip()
        ]
        if not allowed:
            raise ValueError(f"Filter '{item}' does not include any values to match.")
        filtered = filtered[filtered[column].isin(allowed)]
    return filtered


def prepare_metric_series(
    df: pd.DataFrame, spec: PlotSpec, scenario_label: str
) -> pd.DataFrame:
    if spec.column not in df.columns:
        available = ", ".join(sorted(df.columns))
        raise ValueError(
            "The selected plot '{plot}' requires a '{column}' column in the CSV. Available "
            "columns: {available}.".format(
                plot=spec.key, column=spec.column, available=available
            )
        )
    grouped = (
        df.groupby("clients", as_index=False)[spec.column]
        .mean()
        .sort_values("clients")
        .rename(columns={spec.column: "value"})
    )
    if spec.transform is None:
        spec_transform = _identity_transform
    else:
        spec_transform = spec.transform
    return spec_transform(grouped, scenario_label)


def determine_filename_component(
    column: str,
    scenario_info: dict[str, object],
    scenario_df: pd.DataFrame,
    *,
    allow_multiple: bool = False,
) -> str:
    if column in scenario_info and pd.notna(scenario_info[column]):
        return sanitize_for_filename(str(scenario_info[column]))
    if column not in scenario_df.columns:
        raise ValueError(
            "Cannot determine '{column}' for the output filename because the column is "
            "missing from the CSV. Add the column to the dataset or provide the value "
            "via filters.".format(column=column)
        )
    unique_values = scenario_df[column].dropna().unique()
    if len(unique_values) == 1:
        return sanitize_for_filename(str(unique_values[0]))
    if allow_multiple:
        sanitized_values = [
            sanitize_for_filename(str(value)) for value in sorted(unique_values, key=lambda x: str(x))
        ]
        joined = "-".join(sanitized_values)
        return f"multi-{joined}" if joined else "multi"
    raise ValueError(
        "Cannot determine a unique '{column}' value for the output filename. Include "
        "'{column}' in the scenario columns or filter the input so the value is "
        "constant.".format(column=column)
    )



def render_plot(
    spec: PlotSpec,
    scenario_label: str,
    scenario_info: dict[str, object],
    scenario_df: pd.DataFrame,
    series_list: Sequence[tuple[str, pd.DataFrame, object | None]],
    output_dir: Path,
    run_id: str,
    title_override: str | None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(series_list))]

    scale_components: List[str] = []

    for idx, (series_label, data, raw_scale) in enumerate(series_list):
        clients = data["clients"].to_numpy()
        values = data["value"].to_numpy()

        if len(clients) == 0:
            continue

        color = colors[idx]
        legend_label = f"{series_label} - {spec.legend_label}" if spec.legend_label else series_label
        ax.scatter(clients, values, label=legend_label, marker="o", color=color)

        if spec.fit == "usl":
            if len(clients) < 3:
                raise ValueError(
                    f"USL fitting for {series_label} requires at least three client data points."
                )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    params, _ = curve_fit(usl_model, clients, values, bounds=(0, [1, 1]))
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to fit USL curve for {series_label}: {exc}."
                ) from exc
            alpha, beta = params
            print(
                f"[{spec.key}] {series_label}: Estimated α (contention)={alpha:.4f}, β (coherency)={beta:.4f}"
            )
            N_fit = np.linspace(clients.min(), clients.max(), 200)
            Y_fit = usl_model(N_fit, alpha, beta)
            ax.plot(N_fit, Y_fit, label=f"{series_label} - USL model", linestyle="--", color=color)
        else:
            ax.plot(clients, values, linestyle="-", linewidth=1.5, color=color)

        if pd.isna(raw_scale):
            scale_components.append("na")
        else:
            scale_components.append(sanitize_for_filename(str(raw_scale)))

    base_title = title_override or spec.title
    final_title = (
        f"{base_title} ({scenario_label})" if scenario_label and scenario_label != "all data" else base_title
    )
    ax.set_xlabel("Number of Clients (N)")
    ax.set_ylabel(spec.ylabel)
    ax.set_title(final_title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    filename_stub = sanitize_for_filename(spec.filename_stub or spec.key)
    workers_value = determine_filename_component(
        "workers", scenario_info, scenario_df, allow_multiple=True
    )
    unique_scales = []
    for component in scale_components:
        if component not in unique_scales:
            unique_scales.append(component)
    if not unique_scales:
        scale_fragment = "scale-unknown"
    elif len(unique_scales) == 1:
        scale_fragment = f"scale-{unique_scales[0]}"
    else:
        joined = "-".join(unique_scales)
        scale_fragment = f"scales-{joined}"
    output_name = f"{run_id}_{filename_stub}_workers-{workers_value}_{scale_fragment}.png"
    output_path = output_dir / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


EXTRA_SERIES_COLUMNS = ["work", "work_number", "workload"]


def collect_worker_series(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    workers_column: str | None,
    base_label: str,
    raw_scale: object | None,
    extra_series_columns: Sequence[str] | None = None,
) -> List[tuple[str, pd.DataFrame, object | None]]:
    series_list: List[tuple[str, pd.DataFrame, object | None]] = []

    series_columns: List[str] = []
    if workers_column and df[workers_column].notna().any():
        series_columns.append(workers_column)

    if extra_series_columns:
        for column in extra_series_columns:
            if column == workers_column or column in series_columns:
                continue
            if column not in df.columns:
                continue
            if not df[column].notna().any():
                continue
            series_columns.append(column)

    for candidate in EXTRA_SERIES_COLUMNS:
        if candidate == workers_column or candidate not in df.columns:
            continue
        if not df[candidate].notna().any():
            continue
        series_columns.append(candidate)

    if series_columns:
        grouped = df.groupby(series_columns, dropna=False, sort=True)
        for key, subset in grouped:
            if not isinstance(key, tuple):
                key_tuple = (key,)
            else:
                key_tuple = key

            label_parts: List[str] = []
            for column, value in zip(series_columns, key_tuple):
                if pd.isna(value):
                    value_repr = "na"
                else:
                    value_repr = str(value)
                label_parts.append(f"{column}={value_repr}")

            series_label = ", ".join(label_parts) if label_parts else "all data"
            transform_label = (
                f"{base_label}, {series_label}" if base_label else series_label
            )

            metric_df = prepare_metric_series(subset, spec, transform_label)
            if metric_df.empty:
                continue
            series_list.append((series_label, metric_df, raw_scale))
        return series_list

    metric_df = prepare_metric_series(df, spec, base_label or "all data")
    if not metric_df.empty:
        series_list.append(("all data", metric_df, raw_scale))
    return series_list


def main() -> None:
    args = parse_args()
    csv_path = resolve_path(args.csv)
    output_base_dir = resolve_path(args.output_dir)

    df = pd.read_csv(csv_path)

    if "clients" not in df.columns:
        raise ValueError("The input CSV must contain a 'clients' column with concurrency levels.")

    if args.filter:
        df = apply_filters(df, args.filter)

    if df.empty:
        raise ValueError("No rows left after applying the requested filters.")

    series_columns = infer_scenario_columns(df, args.scenario_columns)
    run_id = sanitize_for_filename(args.run_id) if args.run_id else infer_run_identifier(df, csv_path)
    plot_keys = normalize_plots(args.plots)

    output_base_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = output_base_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: List[Path] = []

    scale_column = "scale" if "scale" in df.columns else None
    workers_column = "workers" if "workers" in df.columns else None

    if scale_column:
        scale_series = df[scale_column]
        numeric_series = pd.to_numeric(scale_series, errors="coerce")

        scale_order_df = pd.DataFrame({
            "raw": scale_series,
            "numeric": numeric_series,
        })
        scale_order_df = scale_order_df.drop_duplicates(subset="raw")
        scale_order_df = scale_order_df.sort_values(
            by=["numeric", "raw"],
            kind="mergesort",
        )
        unique_scales: List[object] = scale_order_df["raw"].dropna().tolist()

        if scale_series.isna().any():
            unique_scales.append(None)
    else:
        unique_scales = [None]

    for spec_key in plot_keys:
        spec = PLOT_TYPES[spec_key]
        for raw_scale in unique_scales:
            if scale_column is None:
                scale_df = df
            elif raw_scale is None:
                scale_df = df[df[scale_column].isna()]
            else:
                scale_df = df[df[scale_column] == raw_scale]

            if scale_df.empty:
                continue

            if raw_scale is None and scale_column is None:
                scale_label = "all data"
            elif raw_scale is None:
                scale_label = "scale=unknown"
            else:
                scale_label = f"scale={raw_scale}"

            series_list = collect_worker_series(
                scale_df,
                spec=spec,
                workers_column=workers_column,
                base_label=scale_label,
                raw_scale=raw_scale,
                extra_series_columns=series_columns,
            )

            if not series_list:
                print(f"[{spec.key}] Skipping {scale_label}: no datapoints available.")
                continue

            scenario_info: dict[str, object] = {}
            if scale_column is not None:
                scenario_info[scale_column] = raw_scale

            try:
                output_path = render_plot(
                    spec,
                    scale_label,
                    scenario_info,
                    scale_df,
                    series_list,
                    run_output_dir,
                    run_id,
                    args.title,
                )
            except ValueError as exc:
                print(f"[{spec.key}] Skipping {scale_label}: {exc}")
                continue
            generated_files.append(output_path)
            print(f"[{spec.key}] Saved plot to {output_path}")

    if not generated_files:
        raise RuntimeError("No plots were generated. Check the filters and available data.")


if __name__ == "__main__":
    main()


