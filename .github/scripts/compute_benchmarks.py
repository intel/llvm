#!/usr/bin/env python3

# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess  # nosec B404
import csv
import argparse
import io
import json
from pathlib import Path

# Function to run the benchmark with the given parameters and environment variables
def run_benchmark(directory, ioq, env_vars):
    env = os.environ.copy()
    env.update(env_vars)
    command = [
        f"{directory}/api_overhead_benchmark_sycl",
        "--test=SubmitKernel",
        f"--Ioq={ioq}",
        "--DiscardEvents=0",
        "--MeasureCompletion=0",
        "--iterations=10000",
        "--Profiling=0",
        "--NumKernels=10",
        "--KernelExecTime=1",
        "--csv",
        "--noHeaders"
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)  # nosec B603
    return command, result.stdout

# Function to parse the CSV output and extract the mean execution time
def parse_output(output):
    # Use StringIO to turn the string output into a file-like object for the csv reader
    csv_file = io.StringIO(output)
    reader = csv.reader(csv_file)

    # Skip the header row
    next(reader, None)
    data_row = next(reader, None)
    if data_row is None:
        raise ValueError("Benchmark output does not contain data.")
    try:
        name = data_row[0] # Name of the benchmark is the first value
        mean = float(data_row[1]) # Mean is the second value
        return (name, mean)
    except ValueError:
        raise ValueError(f"Could not convert mean execution time to float: '{data_row[1]}'")
    except IndexError:
        raise ValueError("Data row does not contain enough values.")

# Function to generate the mermaid bar chart script
def generate_mermaid_script(labels, chart_data):
    mermaid_script=f"""
---
config:
    gantt:
        rightPadding: 10
        leftPadding: 120
        sectionFontSize: 10
        numberSectionStyles: 2
---
gantt
    title api_overhead_benchmark_sycl, mean execution time per 10 kernels (μs)
    todayMarker off
    dateFormat  X
    axisFormat %s
"""
    for label in labels:
        nbars = 0
        print_label = label.replace(" ", "<br>")
        mermaid_script += f"""
    section {print_label}
"""
        for (name, data) in chart_data:
            if data is not None:
                if label in data:
                    nbars += 1
                    mean = data[label]
                    crit = "crit," if name == "This PR" else ""
                    mermaid_script += f"""
        {name} ({mean} us)   : {crit} 0, {int(mean)}
"""
        padding = 4 - nbars
        if padding > 0:
            for _ in range(padding):
                mermaid_script += f"""
    -   : 0, 0
"""

    return mermaid_script

# Function to generate the markdown collapsible sections for each variant
def generate_markdown_details(variant_details):
    markdown_sections = []
    for label, command, env_vars, output in variant_details:
        env_vars_str = '\n'.join(f"{key}={value}" for key, value in env_vars.items())
        markdown_sections.append(f"""
<details>
<summary>{label}</summary>

#### Environment Variables:
{env_vars_str}

#### Command:
{' '.join(command)}

#### Output:
{output}

</details>
""")
    return "\n".join(markdown_sections)

# Function to generate the full markdown
def generate_markdown_with_mermaid_chart(mermaid_script, variant_details):
    return f"""
# Benchmark Results
```mermaid
{mermaid_script}
```
## Details
{generate_markdown_details(variant_details)}
"""

def save_benchmark_results(save_name, benchmark_data):
    benchmarks_dir = Path.home() / 'benchmarks'
    benchmarks_dir.mkdir(exist_ok=True)
    file_path = benchmarks_dir / f"{save_name}.json"
    with file_path.open('w') as file:
        json.dump(benchmark_data, file, indent=4)
    print(f"Benchmark results saved to {file_path}")

def load_benchmark_results(compare_name):
    benchmarks_dir = Path.home() / 'benchmarks'
    file_path = benchmarks_dir / f"{compare_name}.json"
    if file_path.exists():
        with file_path.open('r') as file:
            return json.load(file)
    else:
        return None

def main(directory, additional_env_vars, save_name, compare_names):
    variants = [
        (1, {'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '0'}, "Imm-CmdLists-OFF"),
        (0, {'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '0'}, "Imm-CmdLists-OFF"),
        (1, {'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '1'}, ""),
        (0, {'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '1'}, ""),
    ]

    # Run benchmarks and collect means, labels, and variant details
    means = []
    labels = []
    variant_details = []
    for ioq, env_vars, extra_label in variants:
        merged_env_vars = {**env_vars, **additional_env_vars}
        command, output = run_benchmark(directory, ioq, merged_env_vars)
        (label, mean) = parse_output(output)
        label += f" {extra_label}"
        means.append(mean)
        labels.append(label)
        variant_details.append((label, command, merged_env_vars, output))

    benchmark_data = {label: mean for label, mean in zip(labels, means)}

    chart_data = [("This PR", benchmark_data)]
    for name in compare_names:
        chart_data.append((name, load_benchmark_results(name)))

    if save_name:
        save_benchmark_results(save_name, benchmark_data)

    mermaid_script = generate_mermaid_script(labels, chart_data)

    markdown_content = generate_markdown_with_mermaid_chart(mermaid_script, variant_details)

    with open('benchmark_results.md', 'w') as file:
        file.write(markdown_content)

    print("Markdown with benchmark results has been written to benchmark_results.md")

def validate_and_parse_env_args(env_args):
    env_vars = {}
    for arg in env_args:
        if '=' not in arg:
            raise ValueError(f"Environment variable argument '{arg}' is not in the form Variable=Value.")
        key, value = arg.split('=', 1)
        env_vars[key] = value
    return env_vars

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks and generate a Mermaid bar chart script.')
    parser.add_argument('benchmark_directory', type=str, help='The directory where the benchmarks are located.')
    parser.add_argument("--env", type=str, help='Use env variable for a benchmark run.', action="append", default=[])
    parser.add_argument("--save", type=str, help='Save the results for comparison under a specified name.')
    parser.add_argument("--compare", type=str, help='Compare results against previously saved data.', action="append", default=["baseline"])

    args = parser.parse_args()

    additional_env_vars = validate_and_parse_env_args(args.env)

    main(args.benchmark_directory, additional_env_vars, args.save, args.compare)
