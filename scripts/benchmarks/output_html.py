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
from benches.result import BenchmarkRun, Result
import numpy as np

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
class BenchmarkChart:
    label: str
    html: str

def tooltip_css() -> str:
    return '.mpld3-tooltip{background:white;padding:8px;border:1px solid #ddd;border-radius:4px;font-family:monospace;white-space:pre;}'

def create_time_series_chart(benchmarks: list[BenchmarkSeries], github_repo: str) -> list[BenchmarkChart]:
    plt.close('all')

    num_benchmarks = len(benchmarks)
    if num_benchmarks == 0:
        return []

    html_charts = []

    for _, benchmark in enumerate(benchmarks):
        fig, ax = plt.subplots(figsize=(10, 4))

        all_values = []
        all_stddevs = []

        for run in benchmark.runs:
            sorted_points = sorted(run.results, key=lambda x: x.date)
            dates = [point.date for point in sorted_points]
            values = [point.value for point in sorted_points]
            stddevs = [point.stddev for point in sorted_points]

            all_values.extend(values)
            all_stddevs.extend(stddevs)

            ax.errorbar(dates, values, yerr=stddevs, fmt='-', label=run.name, alpha=0.5)
            scatter = ax.scatter(dates, values, picker=True)

            tooltip_labels = [
                f"Date: {point.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Value: {point.value:.2f} {benchmark.metadata.unit}\n"
                f"Stddev: {point.stddev:.2f} {benchmark.metadata.unit}\n"
                f"Git Hash: {point.git_hash}"
                for point in sorted_points
            ]

            targets = [f"https://github.com/{github_repo}/commit/{point.git_hash}"
                      for point in sorted_points]

            tooltip = mpld3.plugins.PointHTMLTooltip(scatter, tooltip_labels,
                css=tooltip_css(),
                targets=targets)
            mpld3.plugins.connect(fig, tooltip)

        # This is so that the stddev doesn't fill the entire y axis on the chart
        if all_values and all_stddevs:
            max_value = max(all_values)
            min_value = min(all_values)
            max_stddev = max(all_stddevs)
            ax.set_ylim(min_value - 3 * max_stddev, max_value + 3 * max_stddev)

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

        plt.tight_layout()
        html_charts.append(BenchmarkChart(html=mpld3.fig_to_html(fig), label=benchmark.label))
        plt.close(fig)

    return html_charts

@dataclass
class ExplicitGroup:
    name: str
    nnames: int
    metadata: BenchmarkMetadata
    runs: dict[str, dict[str, Result]]

def create_explicit_groups(benchmark_runs: list[BenchmarkRun], compare_names: list[str]) -> list[ExplicitGroup]:
    groups = {}

    for run in benchmark_runs:
        if run.name in compare_names:
            for res in run.results:
                if res.explicit_group != '':
                    if res.explicit_group not in groups:
                        groups[res.explicit_group] = ExplicitGroup(name=res.explicit_group, nnames=len(compare_names),
                                metadata=BenchmarkMetadata(unit=res.unit, lower_is_better=res.lower_is_better),
                                runs={})

                    group = groups[res.explicit_group]
                    if res.label not in group.runs:
                        group.runs[res.label] = {name: None for name in compare_names}

                    if group.runs[res.label][run.name] is None:
                        group.runs[res.label][run.name] = res

    return list(groups.values())

def create_grouped_bar_charts(groups: list[ExplicitGroup]) -> list[BenchmarkChart]:
    plt.close('all')

    html_charts = []

    for group in groups:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(group.nnames)
        x_labels = []
        width = 0.8 / len(group.runs)

        max_height = 0

        for i, (run_name, run_results) in enumerate(group.runs.items()):
            offset = width * i

            positions = x + offset
            x_labels = run_results.keys()
            valid_data = [r.value if r is not None else 0 for r in run_results.values()]
            rects = ax.bar(positions, valid_data, width, label=run_name)
            # This is a hack to disable all bar_label. Setting labels to empty doesn't work.
            # We create our own labels below for each bar, this works better in mpld3.
            ax.bar_label(rects, fmt='')

            for rect, run, res in zip(rects, run_results.keys(), run_results.values()):
                height = rect.get_height()
                if height > max_height:
                    max_height = height

                ax.text(rect.get_x() + rect.get_width()/2., height + 1,
                                    f'{res.value:.1f}',
                                    ha='center', va='bottom', fontsize=9)

                tooltip_labels = [
                    f"Date: {res.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Run: {run}\n"
                    f"Label: {res.label}\n"
                    f"Value: {res.value:.2f} {res.unit}\n"
                    f"Stddev: {res.stddev:.2f} {res.unit}\n"
                ]
                tooltip = mpld3.plugins.LineHTMLTooltip(rect, tooltip_labels, css=tooltip_css())
                mpld3.plugins.connect(ax.figure, tooltip)

        # normally we'd just set legend to be outside
        # the chart, but this is not supported by mpld3.
        # instead, we adjust the y axis to account for
        # the height of the bars.
        legend_height = len(group.runs) * 0.1
        ax.set_ylim(0, max_height * (1 + legend_height))

        ax.set_xticks([])
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_ylabel(f"Value ({group.metadata.unit})")
        ax.legend(loc='upper left')
        ax.set_title(group.name, pad=20)
        performance_indicator = "lower is better" if group.metadata.lower_is_better else "higher is better"
        ax.text(0.5, 1.03, f"({performance_indicator})",
                ha='center',
                transform=ax.transAxes,
                style='italic',
                fontsize=7,
                color='#666666')

        for idx, label in enumerate(x_labels):
            # this is a hack to get labels to show above the legend
            # we normalize the idx to transAxes transform and offset it a little.
            x_norm = (idx + 0.3 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
            ax.text(x_norm, 1.03, label,
                transform=ax.transAxes,
                color='#666666')

        plt.tight_layout()
        html_charts.append(BenchmarkChart(label=group.name, html=mpld3.fig_to_html(fig)))
        plt.close(fig)

    return html_charts

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
    benchmarks = process_benchmark_data(benchmark_runs, compare_names)

    timeseries = create_time_series_chart(benchmarks, github_repo)
    timeseries_charts_html = '\n'.join(f'<div class="chart" data-label="{ts.label}"><div>{ts.html}</div></div>' for ts in timeseries)

    explicit_groups = create_explicit_groups(benchmark_runs, compare_names)

    bar_charts = create_grouped_bar_charts(explicit_groups)
    bar_charts_html = '\n'.join(f'<div class="chart" data-label="{bc.label}"><div>{bc.html}</div></div>' for bc in bar_charts)

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
            .filter-container {{
                text-align: center;
                margin-bottom: 24px;
            }}
            .filter-container input {{
                padding: 8px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                width: 400px;
                max-width: 100%;
            }}
            details {{
                margin-bottom: 24px;
            }}
            summary {{
                font-size: 18px;
                font-weight: 500;
                cursor: pointer;
                padding: 12px;
                background: #e9ecef;
                border-radius: 8px;
                user-select: none;
            }}
            summary:hover {{
                background: #dee2e6;
            }}
        </style>
        <script>
            function getQueryParam(param) {{
                const urlParams = new URLSearchParams(window.location.search);
                return urlParams.get(param);
            }}

            function filterCharts() {{
                const regexInput = document.getElementById('bench-filter').value;
                const regex = new RegExp(regexInput, 'i');
                const charts = document.querySelectorAll('.chart');
                let timeseriesVisible = false;
                let barChartsVisible = false;

                charts.forEach(chart => {{
                    const label = chart.getAttribute('data-label');
                    if (regex.test(label)) {{
                        chart.style.display = '';
                        if (chart.closest('.timeseries')) {{
                            timeseriesVisible = true;
                        }} else if (chart.closest('.bar-charts')) {{
                            barChartsVisible = true;
                        }}
                    }} else {{
                        chart.style.display = 'none';
                    }}
                }});

                updateURL(regexInput);

                document.querySelector('.timeseries').open = timeseriesVisible;
                document.querySelector('.bar-charts').open = barChartsVisible;
            }}

            function updateURL(regex) {{
                const url = new URL(window.location);
                if (regex) {{
                    url.searchParams.set('regex', regex);
                }} else {{
                    url.searchParams.delete('regex');
                }}
                history.replaceState(null, '', url);
            }}

            document.addEventListener('DOMContentLoaded', (event) => {{
                const regexParam = getQueryParam('regex');
                if (regexParam) {{
                    document.getElementById('bench-filter').value = regexParam;
                    filterCharts();
                }}
            }});
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Benchmark Results</h1>
            <div class="filter-container">
                <input type="text" id="bench-filter" placeholder="Regex..." oninput="filterCharts()">
            </div>
            <details class="timeseries">
                <summary>Historical Results</summary>
                <div class="charts">
                    {timeseries_charts_html}
                </div>
            </details>
            <details class="bar-charts">
                <summary>Comparisons</summary>
                <div class="charts">
                    {bar_charts_html}
                </div>
            </details>
        </div>
    </body>
    </html>
    """
    return html_template
