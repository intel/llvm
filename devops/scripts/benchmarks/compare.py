import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from utils.aggregate import SimpleMedian
from utils.validate import Validate
from utils.result import Result, BenchmarkRun
from options import options

@dataclass
class BenchmarkHistoricAverage:
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

    @staticmethod
    def get_hist_avg(
        result_name: str, result_dir: str, cutoff: str, aggregator=SimpleMedian,
        exclude: list[str] = []
    ) -> dict[str, BenchmarkHistoricAverage]:

        def get_timestamp(f: str) -> str:
            """Extract timestamp from result filename"""
            return str(f)[-len("YYYYMMDD_HHMMSS.json") : -len(".json")]

        def get_result_paths() -> list[str]:
            """
            Get a list of all results matching result_name in result_dir that is
            newer than the timestamp specified by cutoff
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
                    cache_dir.glob(f"{result_name}_*_*.json")
                )
            )

        # key: name of the benchmark test result
        # value: { command_args: set[str], aggregate: Aggregator }
        # 
        # This is then used to build a dict[BenchmarkHistoricAverage] used
        # to find historic averages.
        average_aggregate: dict[str, dict] = dict()
        
        for result_path in get_result_paths():
            with result_path.open('r') as result_f:
                result = BenchmarkRun.from_json(json.load(result_f))
            
            if result.name != result_name:
                print(f"Warning: Result file {result_path} has mismatching name {result.name}. Skipping file.")
                continue

            for test_run in result.results:
                def reset_aggregate() -> dict:
                    return { 
                        "command_args": set(test_run.command[1:]),
                        # The assumption here is that "value" is median
                        # TODO standardization should happen here on what "value"
                        # really is
                        "aggregate": aggregator(starting_elements=[test_run.value])
                    }

                # Add every benchmark run to average_aggregate:
                if test_run.name not in average_aggregate:
                    average_aggregate[test_run.name] = reset_aggregate()
                else:
                    # Check that we are comparing runs with the same cmd args:
                    if set(test_run.command[1:]) == average_aggregate[test_run.name]["command_args"]:
                        average_aggregate[test_run.name]["aggregate"].add(test_run.value)
                    else:
                        # If the command args used between runs are different,
                        # discard old run data and prefer new command args
                        #
                        # This relies on the fact that paths from get_result_paths()
                        # is sorted from older to newer
                        print(f"Warning: Command args for {test_run.name} from {result_path} is different from prior runs.")
                        print("DISCARDING older data and OVERRIDING with data using new arg.")
                        average_aggregate[test_run.name] = reset_aggregate()
            
        return {
            name: BenchmarkHistoricAverage(
                name=name,
                average_type=stats["aggregate"].get_type(),
                value=stats["aggregate"].get_avg(),
                command_args=stats["command_args"]
            )
            for name, stats in average_aggregate.items()
        }
    

    def to_hist_avg(
        hist_avg: dict[str, BenchmarkHistoricAverage], compare_file: str
    ) -> tuple:
        with open(compare_file, 'r') as compare_f:
            compare_result = BenchmarkRun.from_json(json.load(compare_f))

        improvement = []
        regression = []

        for test in compare_result.results:
            if test.name not in hist_avg:
                continue
            if hist_avg[test.name].command_args != set(test.command[1:]):
                print(f"Warning: skipped {test.name} due to command args mismatch.")
                continue
            
            delta = 1 - (
                test.value / hist_avg[test.name].value
                if test.lower_is_better else 
                hist_avg[test.name].value / test.value
            )

            def perf_diff_entry() -> dict:
                res = asdict(test)
                res["delta"] = delta
                res["hist_avg"] = hist_avg[test.name].value
                res["avg_type"] = hist_avg[test.name].average_type
                return res

            if delta > options.regression_threshold:
                improvement.append(perf_diff_entry())
            elif delta < -options.regression_threshold:
                regression.append(perf_diff_entry())

        return improvement, regression
            



    def to_hist(
        avg_type: str, result_name: str, compare_name: str, result_dir: str, cutoff: str,
        
    ) -> tuple:
        """
        This function generates a historic average from results named result_name
        in result_dir and compares it to the results in compare_file

        Parameters:
            result_name (str): Save name of the result
            compare_name (str): Result file name to compare historic average against
            result_dir (str): Directory to look for results in
            cutoff (str): Timestamp (in YYYYMMDD_HHMMSS) indicating the oldest
            result included in the historic average calculation
            avg_type (str): Type of "average" (measure of central tendency) to 
            use in historic "average" calculation
        """ 

        if avg_type != "median":
            print("Only median is currently supported: refusing to continue.")
            exit(1)

        # TODO call validator on cutoff timestamp
        hist_avg = Compare.get_hist_avg(result_name, result_dir, cutoff, exclude=[compare_name])
        return Compare.to_hist_avg(hist_avg, f"{result_dir}/{compare_name}.json")


res = Compare.to_hist("median", "Baseline_PVC_L0", "Baseline_PVC_L0_20250314_170754", "./", "00000000_000000")
print(res)
