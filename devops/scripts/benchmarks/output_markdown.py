# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
# Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
from utils.result import Result, BenchmarkMetadata
from options import options, MarkdownSize
import ast


class OutputLine:
    def __init__(self, name):
        self.label = name
        self.diff = None
        self.bars = None
        self.row = ""
        self.suite = "Unknown"
        self.explicit_group = ""

    def __str__(self):
        return f"(Label:{self.label}, diff:{self.diff})"

    def __repr__(self):
        return self.__str__()


# The number of the required columns in the markdown table,
# independent of the chart_data content.
# Required columns:
# - benchmark_name
#
# optional +1: relative performance
num_info_columns = 1

# Number of columns required for relative performance change calculation.
# In case of multiple provided saved baselines to compare, the relative
# performance is not calculated, since the base (hopefully) usage case
# for this script would be comparing the performance of PR with the main branch
num_baselines_required_for_rel_change = 2

# Maximum number of characters that is allowed in request validation
# for posting comments in GitHub PRs
max_markdown_size = 65536


def is_relative_perf_comparison_to_be_performed(
    chart_data: dict[str, list[Result]], baseline_name: str
):
    return (len(chart_data) == num_baselines_required_for_rel_change) and (
        baseline_name in chart_data.keys()
    )


def get_chart_markdown_header(chart_data: dict[str, list[Result]], baseline_name: str):
    summary_header = ""
    final_num_columns = num_info_columns

    if is_relative_perf_comparison_to_be_performed(chart_data, baseline_name):
        summary_header = (
            "| Benchmark | " + " | ".join(chart_data.keys()) + " | Change |\n"
        )
        final_num_columns += 1
    else:
        summary_header = "| Benchmark | " + " | ".join(chart_data.keys()) + " |\n"

    summary_header += "|---" * (len(chart_data) + final_num_columns) + "|\n"

    return summary_header


def get_improved_regressed_summary(is_improved: bool, rows_count: int):
    title = "Improved"
    if not is_improved:
        title = "Regressed"

    summary = (
        "\n<details>\n"
        "<summary>\n"
        f"{title} {rows_count} "
        f"(threshold {options.stddev_threshold*100:.2f}%)\n"
        "</summary>\n\n"
    )

    return summary


def get_relative_perf_summary(group_size: int, group_name: str):
    summary = (
        "\n<details>\n"
        f"<summary> Relative perf in group {group_name} "
        f"({group_size})\n"
        "</summary>\n\n"
    )

    return summary


def get_main_branch_run_name(chart_data: dict[str, list[Result]], baseline_name: str):
    for key in chart_data.keys():
        if key != baseline_name:
            return key

    return None


def get_available_markdown_size(current_markdown_size: int):
    return max(0, max_markdown_size - current_markdown_size)


def is_content_in_size_limit(content_size: int, current_markdown_size: int):
    return content_size <= get_available_markdown_size(current_markdown_size)


def get_explicit_group_name(result: Result, metadata: dict[str, BenchmarkMetadata]):
    explicit_group_name = ""
    try:
        explicit_group_name = metadata[result.label].explicit_group
    except Exception as e:
        print(
            f"Warning: Unexpected error when getting explicit_group for '{result.label}': {e}"
        )
        return "Other"

    return explicit_group_name if explicit_group_name else "Other"


# Function to generate the markdown collapsible sections for each variant
def generate_markdown_details(
    results: list[Result], current_markdown_size: int, markdown_size: MarkdownSize
):
    markdown_sections = []
    markdown_start = (
        "\n<details>\n"
        "<summary>Benchmark details - environment, command..."
        "</summary>\n"
    )
    markdown_sections.append(markdown_start)

    for res in results:
        env_dict = res.env
        command = res.command

        section = (
            "\n<details>\n"
            f"<summary>{res.label}</summary>\n\n"
            "#### Command:\n"
            f"{' '.join(command)}\n\n"
        )

        if env_dict:
            env_vars_str = "\n".join(
                f"{key}={value}" for key, value in env_dict.items()
            )
            section += f"#### Environment Variables:\n {env_vars_str}\n"

        section += "\n</details>\n"

        markdown_sections.append(section)

    markdown_sections.append("\n</details>\n")

    full_markdown = "\n".join(markdown_sections)

    if markdown_size == MarkdownSize.FULL:
        return full_markdown
    else:
        if is_content_in_size_limit(len(full_markdown), current_markdown_size):
            return full_markdown
        else:
            return "\nBenchmark details contain too many chars to display\n"


def generate_summary_table(
    chart_data: dict[str, list[Result]],
    baseline_name: str,
    markdown_size: MarkdownSize,
    metadata: dict[str, BenchmarkMetadata],
):
    summary_table = get_chart_markdown_header(
        chart_data=chart_data, baseline_name=baseline_name
    )

    # Collect all benchmarks and their results
    # key: benchmark name,
    # value: dict(run_name : single_result_in_the_given_run)
    benchmark_results = collections.defaultdict(dict)

    # key: run name
    # results: results from different benchmarks collected in the named run
    for key, results in chart_data.items():
        for res in results:
            benchmark_results[res.name][key] = res

    # Generate the table rows
    output_detailed_list = []

    for bname, results in benchmark_results.items():
        oln = OutputLine(bname)
        oln.row = f"| {bname} |"
        best_value = None
        best_key = None

        are_suite_group_assigned = False

        # Determine the best value for the given benchmark, among the results
        # from all saved runs specified by --compare
        # key: run name,
        # res: single result collected in the given run
        for key, res in results.items():
            if not are_suite_group_assigned:
                oln.suite = res.suite
                oln.explicit_group = get_explicit_group_name(res, metadata)

                are_suite_group_assigned = True

            if (
                best_value is None
                or (res.lower_is_better and res.value < best_value)
                or (not res.lower_is_better and res.value > best_value)
            ):
                best_value = res.value
                best_key = key

        # Generate the row with all the results from saved runs specified by
        # --compare,
        # Highlight the best value in the row with data
        if options.verbose:
            print(f"Results: {results}")
        for key in chart_data.keys():
            if key in results:
                intv = results[key].value
                if key == best_key:
                    # Highlight the best value
                    oln.row += f" <ins>{intv:3f}</ins> {results[key].unit} |"
                else:
                    oln.row += f" {intv:.3f} {results[key].unit} |"
            else:
                oln.row += " - |"

        if is_relative_perf_comparison_to_be_performed(chart_data, baseline_name):
            pr_key = baseline_name
            main_key = get_main_branch_run_name(chart_data, baseline_name)

            if (pr_key in results) and (main_key in results):
                pr_val = results[pr_key].value
                main_val = results[main_key].value
                diff = None
                if pr_val != 0 and results[pr_key].lower_is_better:
                    diff = main_val / pr_val
                elif main_val != 0 and not results[pr_key].lower_is_better:
                    diff = pr_val / main_val

                if diff != None:
                    oln.diff = diff

        output_detailed_list.append(oln)

    sorted_detailed_list = sorted(
        output_detailed_list, key=lambda x: (x.diff is not None, x.diff), reverse=True
    )

    diff_values = [oln.diff for oln in sorted_detailed_list if oln.diff is not None]

    improved_rows = []
    regressed_rows = []

    if len(diff_values) > 0:
        for oln in sorted_detailed_list:
            if oln.diff != None:
                delta = oln.diff - 1
                oln.row += f" {delta*100:.2f}%"

                if abs(delta) > options.stddev_threshold:
                    if delta > 0:
                        improved_rows.append(oln.row + " | \n")
                    else:
                        regressed_rows.append(oln.row + " | \n")

            if options.verbose:
                print(oln.row)

            summary_table += oln.row + "\n"
    else:
        for oln in sorted_detailed_list:
            summary_table += oln.row + "\n"

    regressed_rows.reverse()

    is_at_least_one_diff = False
    summary_line = ""

    if len(improved_rows) > 0:
        is_at_least_one_diff = True
        summary_line += get_improved_regressed_summary(
            is_improved=True, rows_count=len(improved_rows)
        )
        summary_line += get_chart_markdown_header(
            chart_data=chart_data, baseline_name=baseline_name
        )

        for row in improved_rows:
            summary_line += row

        summary_line += "\n</details>"

    if len(regressed_rows) > 0:
        is_at_least_one_diff = True
        summary_line += get_improved_regressed_summary(
            is_improved=False, rows_count=len(regressed_rows)
        )

        summary_line += get_chart_markdown_header(
            chart_data=chart_data, baseline_name=baseline_name
        )

        for row in regressed_rows:
            summary_line += row

        summary_line += "\n</details>"

    if not is_at_least_one_diff:
        summary_line = f"No diffs to calculate performance change"

    if options.verbose:
        print(summary_line)

    summary_table = "\n## Performance change in benchmark groups\n"

    grouped_in_suites = collections.defaultdict(lambda: collections.defaultdict(list))
    for oln in output_detailed_list:
        grouped_in_suites[oln.suite][oln.explicit_group].append(oln)

    for suite_name, suite_groups in grouped_in_suites.items():
        summary_table += f"<details><summary>{suite_name}</summary>\n\n"

        for name, outgroup in suite_groups.items():
            outgroup_s = sorted(
                outgroup, key=lambda x: (x.diff is not None, x.diff), reverse=True
            )

            summary_table += get_relative_perf_summary(
                group_size=len(outgroup_s), group_name=name
            )
            summary_table += get_chart_markdown_header(chart_data, baseline_name)

            for oln in outgroup_s:
                summary_table += f"{oln.row}\n"

            summary_table += "\n</details>\n\n"

        summary_table += "</details>"

    if markdown_size == MarkdownSize.FULL:
        return summary_line, summary_table
    else:
        full_content_size = len(summary_table) + len(summary_line)

        if is_content_in_size_limit(
            content_size=full_content_size, current_markdown_size=0
        ):
            return summary_line, summary_table
        else:
            if is_content_in_size_limit(
                content_size=len(summary_line), current_markdown_size=0
            ):
                return summary_line, ""
            else:
                return "\n# Summary\n" "Benchmark output is too large to display\n\n"


def generate_failures_section(failures: dict[str, str]) -> str:
    if not failures:
        return ""

    section = "\n# Failures\n"
    section += "| Name | Failure |\n"
    section += "|---|---|\n"

    for name, failure in failures.items():
        section += f"| {name} | {failure} |\n"

    return section


def generate_markdown(
    name: str,
    chart_data: dict[str, list[Result]],
    failures: dict[str, str],
    markdown_size: MarkdownSize,
    metadata: dict[str, BenchmarkMetadata],
):
    (summary_line, summary_table) = generate_summary_table(
        chart_data, name, markdown_size, metadata
    )

    current_markdown_size = len(summary_line) + len(summary_table)

    generated_markdown = (
        "\n# Summary\n"
        "(<ins>Emphasized values</ins> are the best results)\n"
        f"{summary_line}\n"
        f"{summary_table}\n\n"
    )

    if name in chart_data.keys():
        markdown_details = generate_markdown_details(
            chart_data[name], current_markdown_size, markdown_size
        )
        generated_markdown += "\n# Details\n" f"{markdown_details}\n"

    failures_section = generate_failures_section(failures)

    return failures_section + generated_markdown
