import os
import csv
import sys
from common import Validate, SanitizedConfig


class Compare:

    @staticmethod
    def to_hist_avg(benchmark_name: str, hist_avg_path: str, test_csv_path: str):
        """
        Compare a benchmark test result to the historical average

        @param test_name  Name of the benchmark of results being compared
        @param hist_avg_path  Path to historical average .csv file
        @param test_csv_path  Path to benchmark result .csv file
        """
        hist_avg = dict()  # stores historical median of the test suite of interest

        # Load metrics from historical median being compared against
        with open(hist_avg_path, "r") as avg_csv:
            for stat in csv.DictReader(avg_csv):
                hist_avg[stat["TestCase"]] = {
                    metric: float(stat[metric])
                    for metric in SanitizedConfig.METRICS_TOLERANCES
                }

        status = 0
        failure_counts = {metric: 0 for metric in SanitizedConfig.METRICS_TOLERANCES}
        with open(test_csv_path, "r") as sample_csv:
            # For every test case in our current benchmark test suite:
            for sample in csv.DictReader(sample_csv):
                test = sample["TestCase"]
                # Ignore test cases we haven't profiled before
                if test not in hist_avg:
                    continue
                test_hist_avg = hist_avg[test]

                # Check benchmark test results against historical median
                for metric, threshold in SanitizedConfig.METRICS_TOLERANCES.items():
                    max_tolerated = test_hist_avg[metric] * (1 + threshold)
                    sample_value = Validate.sanitize_stat(sample[metric])
                    if not isinstance(sample_value, float):
                        print(
                            f"Malformatted statistic in {test_csv_path}: "
                            + f"'{sample[metric]}' for {test}."
                        )
                        exit(1)

                    if sample_value > max_tolerated:
                        # Log failure if fail, otherwise proceed as usual
                        print(f"\n-- FAILED {benchmark_name}::{test}")
                        print(
                            f"  {metric}: {sample_value} -- Historic avg. {test_hist_avg[metric]} (max tolerance {threshold*100}%: {max_tolerated})\n"
                        )
                        with open("./artifact/benchmarks_failed.log", "a") as slow_log:
                            slow_log.write(
                                f"-- {benchmark_name}::{test}\n"
                                f"   {metric}: {sample_value} -- Historic avg. {test_hist_avg[metric]} (max tol. {threshold*100}%: {max_tolerated})\n"
                            )
                        status = 1
                        failure_counts[metric] += 1
        if status != 0:
            print(f"Failure counts: {failure_counts}")
        return status


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <path to /devops> <relative path to results directory> <result csv filename>"
        )
        exit(1)

    if not Validate.filepath(sys.argv[1]):
        print(f"Not a valid filepath: {sys.argv[1]}", file=sys.stderr)
        exit(1)
    # If the filepath provided passed filepath validation, then it is clean
    SanitizedConfig.load(sys.argv[1])

    # Both benchmark results git repo and benchmark.sh output are structured
    # like so:
    # /<device_selector>/<runner>/<test name>
    # This relative path is sys.argv[1], while the name of the csv file we are
    # comparing against is sys.argv[2].
    benchmark_name = os.path.basename(sys.argv[2])
    test_csv_path = f"./artifact/failed_tests/{sys.argv[2]}/{sys.argv[3]}"
    median_path = f"./llvm-ci-perf-results/{sys.argv[2]}/{benchmark_name}-median.csv"

    if not os.path.isfile(test_csv_path):
        print("Invalid test file provided: " + test_csv_path)
        exit(1)
    if not os.path.isfile(median_path):
        print(
            f"Median file for benchmark '{benchmark_name}' not found at {median_path}.\n"
            + "Please compute the median using the aggregate workflow."
        )
        exit(1)

    # Compare to median in this case
    exit(Compare.to_hist_avg(benchmark_name, median_path, test_csv_path))
