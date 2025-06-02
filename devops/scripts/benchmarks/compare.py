from utils.aggregate import Aggregator, SimpleMedian
from utils.validate import Validate
from utils.result import Result, BenchmarkRun
from options import options

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkHistoricAverage:
    """Contains historic average information for 1 benchmark"""

    # Name of benchmark as defined in Benchmark class definition
    name: str

    # Measure of central tendency used to compute "average"
    average_type: str
    # TODO replace this with Compare enum?
    # However, compare enum's use in the history is ambiguous, perhaps a new enum
    # should replace both

    # Value recorded from the benchmark
    value: float
    # TODO "value" in compute_benchmark assumes median, what about tracking e.g.
    # standard deviation through this process?

    # Arguments used to call the benchmark executable.
    #
    # This exists to ensure benchmarks called using different arguments are not
    # compared together.
    command_args: set[str]
    # TODO Ensure ONEAPI_DEVICE_SELECTOR? GPU name itself?


class Compare:
    """Class containing logic for comparisons between results"""

    @staticmethod
    def get_hist_avg(
        result_name: str,
        result_dir: str,
        hostname: str,
        cutoff: str,
        aggregator: Aggregator = SimpleMedian,
        exclude: list[str] = [],
    ) -> dict[str, BenchmarkHistoricAverage]:
        """
        Create a historic average for results named result_name in result_dir
        using the specified aggregator

        Args:
            result_name (str): Name of benchmarking result to obtain average for
            result_dir (str): Path to folder containing benchmark results
            cutoff (str): Timestamp in YYYYMMDD_HHMMSS of oldest results used in
            average calcultaion
            hostname (str): Hostname of machine on which results ran on
            aggregator (Aggregator): The aggregator to use for calculating the
            historic average
            exclude (list[str]): List of filenames (only the stem) to exclude
            from average calculation

        Returns:
            A dictionary mapping benchmark names to BenchmarkHistoricAverage
            objects
        """
        if not Validate.timestamp(cutoff):
            raise ValueError("Provided cutoff time is not a proper timestamp.")

        def get_timestamp(f: str) -> str:
            """Extract timestamp from result filename"""
            return str(f)[-len("YYYYMMDD_HHMMSS.json") : -len(".json")]

        def get_result_paths() -> list[str]:
            """
            Get a list of all results matching result_name in result_dir that is
            newer than the timestamp specified by cutoff based off of filename.

            This function assumes filenames of benchmark result files are
            accurate; files returned by this function will be checked a second
            time once their contents are actually loaded.
            """
            cache_dir = Path(f"{result_dir}")

            # List is sorted by filename: given our timestamp format, the
            # timestamps are sorted from oldest to newest
            return sorted(
                filter(
                    lambda f: f.is_file()
                    and Validate.timestamp(get_timestamp(f))
                    and get_timestamp(f) > cutoff
                    # Result file is not excluded
                    and f.stem not in exclude,
                    # Assumes format is <name>_YYYYMMDD_HHMMSS.json
                    cache_dir.glob(f"{result_name}_*_*.json"),
                )
            )

        def validate_benchmark_result(result: BenchmarkRun) -> bool:
            """
            Returns True if result file:
            - Was ran on the target machine/hostname specified
            - Sanity check: ensure metadata are all expected values:
              - Date is truly before cutoff timestamp
              - Name truly matches up with specified result_name
            """
            if result.hostname != hostname:
                return False
            if result.name != result_name:
                print(
                    f"Warning: Result file {result_path} does not match specified result name {result.name}."
                )
                return False
            if result.date < datetime.strptime(cutoff, "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            ):
                return False
            return True

        # key: name of the benchmark test result
        # value: { command_args: set[str], aggregate: Aggregator }
        #
        # This is then used to build a dict[BenchmarkHistoricAverage] used
        # to find historic averages.
        average_aggregate: dict[str, dict] = dict()

        for result_path in get_result_paths():
            with result_path.open("r") as result_f:
                result = BenchmarkRun.from_json(json.load(result_f))

            # Perform another check on result file here, as get_result_paths()
            # only filters out result files via filename, which:
            # - does not contain enough information to filter out results, i.e.
            #   no hostname information.
            # - information in filename may be mismatched from metadata.
            if not validate_benchmark_result(result):
                continue

            for test_run in result.results:

                def reset_aggregate() -> dict:
                    return {
                        # TODO compare determine which command args have an
                        # impact on perf results, and do not compare arg results
                        # are incomparable
                        "command_args": set(test_run.command[1:]),
                        "aggregate": aggregator(starting_elements=[test_run.value]),
                    }

                # Add every benchmark run to average_aggregate:
                if test_run.name not in average_aggregate:
                    average_aggregate[test_run.name] = reset_aggregate()
                else:
                    average_aggregate[test_run.name]["aggregate"].add(test_run.value)

        return {
            name: BenchmarkHistoricAverage(
                name=name,
                average_type=stats["aggregate"].get_type(),
                value=stats["aggregate"].get_avg(),
                command_args=stats["command_args"],
            )
            for name, stats in average_aggregate.items()
        }

    def to_hist_avg(
        hist_avg: dict[str, BenchmarkHistoricAverage], target: BenchmarkRun
    ) -> tuple:
        """
        Compare results in target to a pre-existing map of historic average.

        Caution: Ensure the generated hist_avg is for results running on the
        same host as target.hostname.

        Args:
            hist_avg (dict): A historic average map generated from get_hist_avg
            target (BenchmarkRun): results to compare against hist_avg

        Returns:
            A tuple returning (list of improved tests, list of regressed tests).
        """

        def halfway_round(value: int, n: int):
            """
            Python's default round() does banker's rounding, which doesn't
            make much sense here. This rounds 0.5 to 1, and -0.5 to -1
            """
            if value == 0:
                return 0
            return int(value * 10**n + 0.5 * (value / abs(value))) / 10**n

        improvement = []
        regression = []

        for test in target.results:
            if test.name not in hist_avg:
                continue
            # TODO compare command args which have an impact on performance
            # (i.e. ignore --save-name): if command results are incomparable,
            # skip the result.

            delta = 1 - (
                test.value / hist_avg[test.name].value
                if test.lower_is_better
                else hist_avg[test.name].value / test.value
            )

            def perf_diff_entry() -> dict:
                res = asdict(test)
                res["delta"] = delta
                res["hist_avg"] = hist_avg[test.name].value
                res["avg_type"] = hist_avg[test.name].average_type
                return res

            # Round to 2 decimal places: not going to fail a test on 0.001% over
            # regression threshold
            if halfway_round(delta, 2) > options.regression_threshold:
                improvement.append(perf_diff_entry())
            elif halfway_round(delta, 2) < -options.regression_threshold:
                regression.append(perf_diff_entry())

        return improvement, regression

    def to_hist(
        avg_type: str,
        result_name: str,
        compare_file: str,
        result_dir: str,
        cutoff: str,
    ) -> tuple:
        """
        Pregenerate a historic average from results named result_name in
        result_dir, and compares the results in compare_file to it

        Args:
            result_name (str): Save name of the result
            compare_name (str): Result file name to compare historic average against
            result_dir (str): Directory to look for results in
            cutoff (str): Timestamp (in YYYYMMDD_HHMMSS) indicating the oldest
            result included in the historic average calculation
            avg_type (str): Type of "average" (measure of central tendency) to
            use in historic "average" calculation

        Returns:
            A tuple returning (list of improved tests, list of regressed tests).
            Each element in each list is a BenchmarkRun object with a hist_avg,
            avg_type, and delta field added, indicating the historic average,
            type of central tendency used for historic average, and the delta
            from the average for this benchmark run.
        """

        if avg_type == "median":
            aggregator_type = SimpleMedian
        elif avg_type == "EWMA":
            aggregator_type = EWMA
        else:
            print("Error: Unsupported avg_type f{avg_type}.")
            exit(1)

        try:
            with open(compare_file, "r") as compare_f:
                compare_result = BenchmarkRun.from_json(json.load(compare_f))
        except:
            print(f"Unable to open {compare_file}.")
            exit(1)

        # Sanity checks:
        if compare_result.hostname == "Unknown":
            print(
                "Hostname for results in {compare_file} unknown, unable to build a historic average: Refusing to continue."
            )
            exit(1)
        if not Validate.timestamp(cutoff):
            print("Invalid timestamp provided, please follow YYYYMMDD_HHMMSS.")
            exit(1)

        # Build historic average and compare results against historic average:
        hist_avg = Compare.get_hist_avg(
            result_name,
            result_dir,
            compare_result.hostname,
            cutoff,
            aggregator=aggregator_type,
            exclude=[Path(compare_file).stem],
        )
        return Compare.to_hist_avg(hist_avg, compare_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    parser_avg = subparsers.add_parser(
        "to_hist", help="Compare a benchmark result to historic average"
    )
    parser_avg.add_argument(
        "--avg-type",
        type=str,
        help="Measure of central tendency to use when computing historic average",
        default="median",
    )
    parser_avg.add_argument(
        "--name",
        type=str,
        required=True,
        help="Save name of the benchmark results to compare to",
    )
    parser_avg.add_argument(
        "--compare-file",
        type=str,
        required=True,
        help="Result file to compare against te historic average",
    )
    parser_avg.add_argument(
        "--results-dir", type=str, required=True, help="Directory storing results"
    )
    parser_avg.add_argument(
        "--cutoff",
        type=str,
        help="Timestamp (in YYYYMMDD_HHMMSS) of oldest result to include in historic average calculation",
        default="20000101_010101",
    )

    args = parser.parse_args()

    if args.operation == "to_hist":
        if not Validate.timestamp(args.cutoff):
            raise ValueError("Timestamp must be provided as YYYYMMDD_HHMMSS.")
        if args.avg_type not in ["median", "EWMA"]:
            print("Only median is currently supported: exiting.")
            exit(1)

        improvements, regressions = Compare.to_hist(
            args.avg_type, args.name, args.compare_file, args.results_dir, args.cutoff
        )

        def print_regression(entry: dict):
            """Print an entry outputted from Compare.to_hist"""
            print(f"Test: {entry['name']}")
            print(f"-- Historic {entry['avg_type']}: {entry['hist_avg']}")
            print(f"-- Run result: {test['value']}")
            print(f"-- Delta: {test['delta']}")
            print("")

        if improvements:
            print("#\n# Improvements:\n#\n")
            for test in improvements:
                print_regression(test)
        if regressions:
            print("#\n# Regressions:\n#\n")
            for test in regressions:
                print_regression(test)
            exit(1)  # Exit 1 to trigger github test failure
        print("\nNo regressions found!")
    else:
        print("Unsupported operation: exiting.")
        exit(1)
