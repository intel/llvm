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
from string import Template

@dataclass
class BenchmarkMetadata:
    unit: str
    suite: str
    lower_is_better: bool

@dataclass
class BenchmarkSeries:
    label: str
    metadata: BenchmarkMetadata
    runs: list[BenchmarkRun]

@dataclass
class BenchmarkChart:
    label: str
    suite: str
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
        html_charts.append(BenchmarkChart(html=mpld3.fig_to_html(fig), label=benchmark.label, suite=benchmark.metadata.suite))
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
                                metadata=BenchmarkMetadata(unit=res.unit, lower_is_better=res.lower_is_better, suite=res.suite),
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
                if res is None:
                    continue

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
        html_charts.append(BenchmarkChart(label=group.name, html=mpld3.fig_to_html(fig), suite=group.metadata.suite))
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
                    lower_is_better=result.lower_is_better,
                    suite=result.suite
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
    timeseries_charts_html = '\n'.join(f'<div class="chart" data-label="{ts.label}" data-suite="{ts.suite}"><div>{ts.html}</div></div>' for ts in timeseries)

    explicit_groups = create_explicit_groups(benchmark_runs, compare_names)

    bar_charts = create_grouped_bar_charts(explicit_groups)
    bar_charts_html = '\n'.join(f'<div class="chart" data-label="{bc.label}" data-suite="{bc.suite}"><div>{bc.html}</div></div>' for bc in bar_charts)

    suite_names = {t.suite for t in timeseries}
    suite_checkboxes_html = ' '.join(f'<label><input type="checkbox" class="suite-checkbox" data-suite="{suite}" checked> {suite}</label>' for suite in suite_names)

    with open('benchmark_results.html.template', 'r') as file:
        html_template = file.read()

    template = Template(html_template)
    data = {
        'suite_checkboxes_html': suite_checkboxes_html,
        'timeseries_charts_html': timeseries_charts_html,
        'bar_charts_html': bar_charts_html,
    }

    return template.substitute(data)
