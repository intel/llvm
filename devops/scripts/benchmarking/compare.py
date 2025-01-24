import os
import csv
import sys
from pathlib import Path

import common

def compare_to_median(test_name: str, median_path: str, test_csv_path: str):
    median = dict() # stores actual median of current testcase
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
            test_case = sample["TestCase"]

            # Ignore test cases we haven't profiled before
            if test_case not in median:
                continue
            hist_median = median[test_case]
            for metric, threshold in common.metrics_variance.items():
                max_tolerated = hist_median[metric] * (1 + threshold)
                sample_value = common.sanitize(sample[metric])
                if sample_value > max_tolerated:
                    print("vvv FAILED vvv")
                    print(test_case)
                    print(
                        f"{metric}: {sample_value} -- Historic avg. {hist_median[metric]} (max tolerance {threshold*100}%: {max_tolerated})"
                    )
                    print("^^^^^^^^^^^^^^")
                    with open(common.BENCHMARK_SLOW_LOG, "a") as slow_log:
                        slow_log.write(
                            f"-- {test_name}::{test_case}\n"
                            f"   {metric}: {sample_value} -- Historic avg. {hist_median[metric]} (max tol. {threshold*100}%: {max_tolerated})\n"
                        )
                    status = 1
                    failure_counts[metric] += 1
    if status != 0:
        print(f"Failure counts: {failure_counts}")
    return status


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <relative path of test results directory> <result csv filename>")
        exit(1)
    # Both benchmark results git repo and benchmark.sh output are structured
    # like so:
    # /<device_selector>/<runner>/<test name>
    # This relative path is sys.argv[1], while the name of the csv file we are
    # comparing against is sys.argv[2].
    common.load_configs()
    test_name = os.path.basename(sys.argv[1])
    test_csv_path = f"{common.OUTPUT_CACHE}/{sys.argv[1]}/{sys.argv[2]}"
    median_path = f"{common.PERF_RES_PATH}/{sys.argv[1]}/{test_name}-median.csv"

    if not os.path.isfile(test_csv_path):
        print("Invalid test file provided: " + test_csv_path)
        exit(1)
    if not os.path.isfile(median_path):
        print(
            f"Median file for test {test_name} not found at {median_path}.\n"
            + "Please calculate the median using the aggregate workflow."
        )
        exit(1)

    exit(compare_to_median(test_name, median_path, test_csv_path))
