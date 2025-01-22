import os
import csv
import sys
from pathlib import Path

import common


# TODO compare_to(metric) instead?
def compare_to_median(runner: str, test_name: str, test_csv_path: str):
    median_path = f"{common.PERF_RES_PATH}/{runner}/{test_name}/{test_name}-median.csv"

    if not os.path.isfile(test_csv_path):
        print("Invalid test file provided: " + test_csv_path)
        exit(-1)
    if not os.path.isfile(median_path):
        print(
            f"Median file for test {test_name} not found at {median_path}.\n"
            + "Please build the median using the aggregate workflow."
        )
        exit(-1)

    median = dict()
    with open(median_path, "r") as median_csv:
        for stat in csv.DictReader(median_csv):
            median[stat["TestCase"]] = {
                metric: float(stat[metric]) for metric in common.metrics_variance
            }

    # TODO read status codes from a config file instead?
    status = 0
    failure_counts = {metric: 0 for metric in common.metrics_variance}
    with open(test_csv_path, "r") as sample_csv:
        for sample in csv.DictReader(sample_csv):
            # Ignore test cases we haven't profiled before
            if sample["TestCase"] not in median:
                continue
            test_median = median[sample["TestCase"]]
            for metric, threshold in common.metrics_variance.items():
                max_tolerated = test_median[metric] * (1 + threshold)
                if common.sanitize(sample[metric]) > max_tolerated:
                    print("vvv FAILED vvv")
                    print(sample["TestCase"])
                    print(
                        f"{metric}: {common.sanitize(sample[metric])} -- Historic avg. {test_median[metric]} (max tolerance {threshold*100}%: {max_tolerated})"
                    )
                    print("^^^^^^^^^^^^^^")
                    with open(common.BENCHMARK_SLOW_LOG, "a") as slow_log:
                        slow_log.write(
                            f"-- {test_name}::{sample['TestCase']}\n"
                            f"   {metric}: {common.sanitize(sample[metric])} -- Historic avg. {test_median[metric]} (max tol. {threshold*100}%: {max_tolerated})\n"
                        )
                    status = 1
                    failure_counts[metric] += 1
    if status != 0:
        print(f"Failure counts: {failure_counts}")
    return status


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <runner name> <test name> <test csv path>")
        exit(-1)
    common.load_configs()
    exit(compare_to_median(sys.argv[1], sys.argv[2], sys.argv[3]))
