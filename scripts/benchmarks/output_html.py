# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import matplotlib.pyplot as plt
import mpld3
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.dates as mdates
import numpy as np
from benches.result import BenchmarkRun, Result

@dataclass
class BenchmarkMetadata:
    unit: str
    lower_is_better: bool

@dataclass
class BenchmarkSeries:
    label: str
    metadata: BenchmarkMetadata
    runs: list[BenchmarkRun]

@dataclass
class LatestResults:
    benchmark_label: str
    run_values: dict[str, float]

    @classmethod
    def from_dict(cls, label: str, values: dict[str, float]) -> 'LatestResults':
        return cls(benchmark_label=label, run_values=values)

def get_latest_results(benchmarks: list[BenchmarkSeries]) -> dict[str, LatestResults]:
    latest_results: dict[str, LatestResults] = {}
    for benchmark in benchmarks:
        run_values = {
            run.name: max(run.results, key=lambda x: x.date).value
            for run in benchmark.runs
        }
        latest_results[benchmark.label] = LatestResults.from_dict(benchmark.label, run_values)
    return latest_results

def prepare_normalized_data(latest_results: dict[str, LatestResults], 
                          benchmarks: list[BenchmarkSeries],
                          group_benchmarks: list[str],
                          non_baseline_runs: list[str],
                          baseline_name: str) -> list[list[float]]:
    normalized_data = []
    benchmark_map = {b.label: b for b in benchmarks}

    for run_name in non_baseline_runs:
        run_data: list[float] = []
        for benchmark_label in group_benchmarks:
            benchmark_data = latest_results[benchmark_label].run_values
            if run_name not in benchmark_data or baseline_name not in benchmark_data:
                run_data.append(None)
                continue

            baseline_value = benchmark_data[baseline_name]
            current_value = benchmark_data[run_name]

            normalized_value = ((baseline_value / current_value) if benchmark_map[benchmark_label].metadata.lower_is_better
                              else (current_value / baseline_value)) * 100
            run_data.append(normalized_value)
        normalized_data.append(run_data)
    return normalized_data

def format_benchmark_label(label: str) -> list[str]:
    words = re.split(' |_', label)
    lines = []
    current_line = []

    # max line length 30
    for word in words:
        if len(' '.join(current_line + [word])) > 30:
            lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)

    if current_line:
        lines.append(' '.join(current_line))

    return lines

def create_bar_plot(ax: plt.Axes,
                   normalized_data: list[list[float]],
                   group_benchmarks: list[str],
                   non_baseline_runs: list[str],
                   latest_results: dict[str, LatestResults],
                   benchmarks: list[BenchmarkSeries],
                   baseline_name: str) -> float:
    x = np.arange(len(group_benchmarks))
    width = 0.8 / len(non_baseline_runs)
    max_height = 0
    benchmark_map = {b.label: b for b in benchmarks}

    for i, (run_name, run_data) in enumerate(zip(non_baseline_runs, normalized_data)):
        offset = width * i - width * (len(non_baseline_runs) - 1) / 2
        positions = x + offset
        valid_data = [v if v is not None else 0 for v in run_data]
        rects = ax.bar(positions, valid_data, width, label=run_name)

        for rect, value, benchmark_label in zip(rects, run_data, group_benchmarks):
            if value is not None:
                height = rect.get_height()
                if height > max_height:
                    max_height = height

                ax.text(rect.get_x() + rect.get_width()/2., height + 2,
                       f'{value:.1f}%',
                       ha='center', va='bottom')

                benchmark_data = latest_results[benchmark_label].run_values
                baseline_value = benchmark_data[baseline_name]
                current_value = benchmark_data[run_name]
                unit = benchmark_map[benchmark_label].metadata.unit

                tooltip_labels = [
                    f"Run: {run_name}\n"
                    f"Value: {current_value:.2f} {unit}\n"
                    f"Normalized to ({baseline_name}): {baseline_value:.2f} {unit}\n"
                    f"Normalized: {value:.1f}%"
                ]
                tooltip = mpld3.plugins.LineHTMLTooltip(rect, tooltip_labels, css='.mpld3-tooltip{background:white;padding:8px;border:1px solid #ddd;border-radius:4px;font-family:monospace;white-space:pre;}')
                mpld3.plugins.connect(ax.figure, tooltip)

    return max_height

def add_chart_elements(ax: plt.Axes,
                      group_benchmarks: list[str],
                      group_name: str,
                      max_height: float) -> None:
    top_padding = max_height * 0.2
    ax.set_ylim(0, max_height + top_padding)
    ax.set_ylabel('Performance relative to baseline (%)')
    ax.set_title(f'Performance Comparison (Normalized to Baseline) - {group_name} Group')
    ax.set_xticks([])

    for idx, label in enumerate(group_benchmarks):
        split_labels = format_benchmark_label(label)
        for i, sublabel in enumerate(split_labels):
            y_pos = max_height + (top_padding * 0.5) + 2 - (i * top_padding * 0.15)
            ax.text(idx, y_pos, sublabel,
                   ha='center',
                   style='italic',
                   color='#666666')

    ax.grid(True, axis='y', alpha=0.2)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

def split_large_groups(benchmark_groups):
    miscellaneous = []
    new_groups = defaultdict(list)

    split_happened = False
    for group, labels in benchmark_groups.items():
        if len(labels) == 1:
            miscellaneous.extend(labels)
        elif len(labels) > 5:
            split_happened = True
            mid = len(labels) // 2
            new_groups[group] = labels[:mid]
            new_groups[group + '_'] = labels[mid:]
        else:
            new_groups[group] = labels

    if miscellaneous:
        new_groups['Miscellaneous'] = miscellaneous

    if split_happened:
        return split_large_groups(new_groups)
    else:
        return new_groups

def group_benchmark_labels(benchmark_labels):
    benchmark_groups = defaultdict(list)
    for label in benchmark_labels:
        group = re.match(r'^[^_\s]+', label)[0]
        benchmark_groups[group].append(label)
    return split_large_groups(benchmark_groups)

def create_normalized_bar_chart(benchmarks: list[BenchmarkSeries], baseline_name: str) -> list[str]:
    latest_results = get_latest_results(benchmarks)

    run_names = sorted(list(set(
        name for result in latest_results.values()
        for name in result.run_values.keys()
    )))

    if baseline_name not in run_names:
        return []

    benchmark_labels = [b.label for b in benchmarks]

    benchmark_groups = group_benchmark_labels(benchmark_labels)

    html_charts = []

    for group_name, group_benchmarks in benchmark_groups.items():
        plt.close('all')
        non_baseline_runs = [n for n in run_names if n != baseline_name]

        if len(non_baseline_runs) == 0:
            continue

        normalized_data = prepare_normalized_data(
            latest_results, benchmarks, group_benchmarks,
            non_baseline_runs, baseline_name
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        max_height = create_bar_plot(
            ax, normalized_data, group_benchmarks, non_baseline_runs,
            latest_results, benchmarks, baseline_name
        )
        add_chart_elements(ax, group_benchmarks, group_name, max_height)

        plt.tight_layout()
        html_charts.append(mpld3.fig_to_html(fig))
        plt.close(fig)

    return html_charts

def create_time_series_chart(benchmarks: list[BenchmarkSeries], github_repo: str) -> str:
    plt.close('all')

    num_benchmarks = len(benchmarks)
    if num_benchmarks == 0:
        return

    fig, axes = plt.subplots(num_benchmarks, 1, figsize=(10, max(4 * num_benchmarks, 30)))

    if num_benchmarks == 1:
        axes = [axes]

    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]

        for run in benchmark.runs:
            sorted_points = sorted(run.results, key=lambda x: x.date)
            dates = [point.date for point in sorted_points]
            values = [point.value for point in sorted_points]

            ax.plot_date(dates, values, '-', label=run.name, alpha=0.5)
            scatter = ax.scatter(dates, values, picker=True)

            tooltip_labels = [
                f"Date: {point.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Value: {point.value:.2f}\n"
                f"Git Hash: {point.git_hash}"
                for point in sorted_points
            ]

            targets = [f"https://github.com/{github_repo}/commit/{point.git_hash}"
                      for point in sorted_points]

            tooltip = mpld3.plugins.PointHTMLTooltip(scatter, tooltip_labels,
                css='.mpld3-tooltip{background:white;padding:8px;border:1px solid #ddd;border-radius:4px;font-family:monospace;white-space:pre;}',
                targets=targets)
            mpld3.plugins.connect(fig, tooltip)

        ax.set_title(benchmark.label, pad=20)
        performance_indicator = "lower is better" if benchmark.metadata.lower_is_better else "higher is better"
        ax.text(0.5, 1.05, f"({performance_indicator})",
                ha='center',
                transform=ax.transAxes,
                style='italic',
                fontsize=7,
                color='#666666')

        ax.set_xlabel('')
        unit = benchmark.metadata.unit
        ax.set_ylabel(f"Value ({unit})" if unit else "Value")
        ax.grid(True, alpha=0.2)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter('%Y-%m-%d %H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    html = mpld3.fig_to_html(fig)

    plt.close(fig)
    return html

def process_benchmark_data(benchmark_runs: list[BenchmarkRun], compare_names: list[str]) -> list[BenchmarkSeries]:
    benchmark_metadata: dict[str, BenchmarkMetadata] = {}
    run_map: dict[str, dict[str, list[Result]]] = defaultdict(lambda: defaultdict(list))

    for run in benchmark_runs:
        if run.name not in compare_names:
            continue

        for result in run.results:
            if result.label not in benchmark_metadata:
                benchmark_metadata[result.label] = BenchmarkMetadata(
                    unit=result.unit,
                    lower_is_better=result.lower_is_better
                )

            result.date = run.date
            result.git_hash = run.git_hash
            run_map[result.label][run.name].append(result)

    benchmark_series = []
    for label, metadata in benchmark_metadata.items():
        runs = [
            BenchmarkRun(name=run_name, results=results)
            for run_name, results in run_map[label].items()
        ]
        benchmark_series.append(BenchmarkSeries(
            label=label,
            metadata=metadata,
            runs=runs
        ))

    return benchmark_series

def generate_html(benchmark_runs: list[BenchmarkRun], github_repo: str, compare_names: list[str]) -> str:
    baseline_name = compare_names[0]
    benchmarks = process_benchmark_data(benchmark_runs, compare_names)

    comparison_html_charts = create_normalized_bar_chart(benchmarks, baseline_name)
    timeseries_html = create_time_series_chart(benchmarks, github_repo)
    comparison_charts_html = '\n'.join(f'<div class="chart"><div>{chart}</div></div>' for chart in comparison_html_charts)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Benchmark Results</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0;
                padding: 16px;
                background: #f8f9fa;
            }}
            .container {{
                max-width: 1100px;
                margin: 0 auto;
            }}
            h1, h2 {{
                color: #212529;
                text-align: center;
                margin-bottom: 24px;
                font-weight: 500;
            }}
            .chart {{
                background: white;
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                overflow-x: auto;
            }}
            .chart > div {{
                min-width: 600px;
                margin: 0 auto;
            }}
            @media (max-width: 768px) {{
                body {{
                    padding: 12px;
                }}
                .chart {{
                    padding: 16px;
                    border-radius: 6px;
                }}
                h1 {{
                    font-size: 24px;
                    margin-bottom: 16px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Benchmark Results</h1>
            <h2>Latest Results Comparison</h2>
            <div class="chart">
                {comparison_charts_html}
            </div>
            <h2>Historical Results</h2>
            <div class="chart">
                {timeseries_html}
            </div>
        </div>
    </body>
    </html>
    """

    return html_template
