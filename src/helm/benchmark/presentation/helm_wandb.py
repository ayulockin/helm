import os
import json
import wandb
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable, Union, Mapping, Tuple, Set

from helm.common.hierarchical_logger import hlog
from helm.benchmark.presentation.summarize import AGGREGATE_WIN_RATE_COLUMN


@dataclass
class Column:
    """Values and metadata for each column of the table."""

    name: str
    group: str
    metric: str
    values: np.ndarray
    lower_is_better: Optional[bool]
    run_spec_names: Optional[List[str]] = None


@dataclass
class Table:
    """Column-based representation of a standard run-group table. See summarize.py for exact documentation."""

    adapters: List[str]
    columns: List[Column]
    mean_win_rates: Optional[np.ndarray] = None


def parse_table(raw_table: Dict[str, Any]) -> Table:
    """Convert raw table dict to a Table. Ignores strongly contaminated table entries."""

    def get_cell_values(cells: List[dict]) -> List[Any]:
        values = []
        for cell in cells:
            value = cell["value"] if "value" in cell else np.nan
            if "contamination_level" in cell and cell["contamination_level"] == "strong":
                value = np.nan
            values.append(value)
        return values

    def get_cell_run_specs(cells: List[dict]) -> List[str]:
        run_specs = []
        for cell in cells:
            run_spec = cell["run_spec_names"] if "run_spec_names" in cell else np.nan
            run_specs.append(run_spec)
        return run_specs

    adapters: Optional[List[str]] = None
    columns: List[Column] = []
    mean_win_rates: Optional[np.ndarray] = None
    for column_index, (header_cell, *column_cells) in enumerate(zip(raw_table["header"], *raw_table["rows"])):
        cell_values = get_cell_values(column_cells)
        if column_index == 0:
            adapters = cell_values
        elif column_index == AGGREGATE_WIN_RATE_COLUMN and "win rate" in header_cell["value"]:
            mean_win_rates = np.array(cell_values)
        else:
            assert "metadata" in header_cell
            name = header_cell["value"]
            group = header_cell["metadata"]["run_group"]
            metric = header_cell["metadata"]["metric"]
            lower_is_better = header_cell["lower_is_better"] if "lower_is_better" in header_cell else None
            run_spec_names = get_cell_run_specs(column_cells)
            columns.append(Column(name, group, metric, np.array(cell_values), lower_is_better, run_spec_names))
    assert adapters is not None

    return Table(adapters, columns, mean_win_rates)


def get_group_tables(base_path: str, group_name: str) -> Dict[str, Table]:
    """Reads and parses group tables. Uses _tables_cache to avoid reprocessing the same table multiple times."""
    with open(os.path.join(base_path, "groups", f"{group_name}.json")) as fp:
        tables = json.load(fp)

    name_to_table: Dict[str, Table] = {}
    for table in tables:
        name_to_table[table["title"]] = parse_table(table)

    return name_to_table


def parse_accuracy_for_wandb(tables: Dict[str, Table]) -> pd.DataFrame:
    """Parses the accuracy table for W&B."""
    accuracy_table = tables["Accuracy"]
    adapters: List[str] = accuracy_table.adapters
    columns: List[Column] = accuracy_table.columns
    mean_win_rates: Union[np.ndarray, None] = accuracy_table.mean_win_rates
    mean_win_rates_iterable = mean_win_rates.tolist() if mean_win_rates is not None else [np.nan] * len(adapters)

    default_pd_columns = ["Model", "Mean win rate"]
    pd_columns = default_pd_columns + [column.name for column in columns]

    data = []
    for adapter, mean_win_rate, *column_values in zip(
        adapters, mean_win_rates_iterable, *[column.values for column in columns]
    ):
        data.append([adapter, mean_win_rate] + column_values)
    df = pd.DataFrame(data, columns=pd_columns)
    return df


def parse_efficiency_for_wandb(tables: Dict[str, Table]) -> pd.DataFrame:
    """Parses the efficiency table for W&B."""
    efficiency_table = tables["Efficiency"]
    adapters: List[str] = efficiency_table.adapters
    columns: List[Column] = efficiency_table.columns
    mean_win_rates: Union[np.ndarray, None] = efficiency_table.mean_win_rates
    mean_win_rates_iterable = mean_win_rates.tolist() if mean_win_rates is not None else [np.nan] * len(adapters)

    default_pd_columns = ["Model", "Mean win rate"]
    pd_columns = default_pd_columns + [column.name for column in columns]
    data = []
    for adapter, mean_win_rate, *column_values in zip(
        adapters, mean_win_rates_iterable, *[column.values for column in columns]
    ):
        data.append([adapter, mean_win_rate] + column_values)
    df = pd.DataFrame(data, columns=pd_columns)
    return df


def main():
    """
    This script creates the plots used in the HELM paper (https://arxiv.org/abs/2211.09110).
    It should be run _after_ running `summarize.py` with the same `benchmark_output` and `suite` arguments and through
    the top-level command `...`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str, help="Path to benchmarking output", default="benchmark_output")
    parser.add_argument("--suite", type=str, help="Name of the suite that we are plotting", required=True)
    parser.add_argument("--plot-format", help="Format for saving plots", default="png", choices=["png", "pdf"])
    args = parser.parse_args()
    base_path = os.path.join(args.output_path, "runs", args.suite)
    if not os.path.exists(os.path.join(base_path, "groups")):
        hlog(f"ERROR: Could not find `groups` directory under {base_path}. Did you run `summarize.py` first?")
        return

    run = wandb.init(entity="wandb", project="de-llm-leaderboard", job_type="helm_eval")

    tables = get_group_tables(base_path, "multilingual_scenarios")

    accuracy_df = parse_accuracy_for_wandb(tables)
    efficiency_df = parse_efficiency_for_wandb(tables)

    run.log(
        {"accuracy_table": wandb.Table(dataframe=accuracy_df), "efficiency_table": wandb.Table(dataframe=efficiency_df)}
    )


if __name__ == "__main__":
    main()
