import os
import re
import ast

# Globals definition
PERF_RES_PATH, metrics_variance, metrics_recorded = None, None, None
BENCHMARK_SLOW_LOG, BENCHMARK_ERROR_LOG = None, None


def sanitize(stat: str) -> float:
    # Get rid of %
    if stat[-1] == "%":
        stat = stat[:-1]
    return float(stat)


def load_configs():
    BENCHMARKING_ROOT = os.getenv("BENCHMARKING_ROOT")
    if BENCHMARKING_ROOT is None:
        # Try to predict where BENCHMARKING_ROOT is based on executable
        BENCHMARKING_ROOT = os.path.dirname(os.path.abspath(__file__))

    benchmarking_ci_conf_path = f"{BENCHMARKING_ROOT}/benchmark-ci.conf"
    if not os.path.isfile(benchmarking_ci_conf_path):
        raise Exception(f"Please provide path to a valid BENCHMARKING_ROOT.")

    global PERF_RES_PATH, OUTPUT_PATH, metrics_variance, metrics_recorded
    global BENCHMARK_ERROR_LOG, BENCHMARK_SLOW_LOG
    perf_res_re = re.compile(r"^PERF_RES_PATH=(.*)$", re.M)
    output_path_re = re.compile(r"^OUTPUT_PATH=(.*)$", re.M)
    m_variance_re = re.compile(r"^METRICS_VARIANCE=(.*)$", re.M)
    m_recorded_re = re.compile(r"^METRICS_RECORDED=(.*)$", re.M)
    b_slow_re = re.compile(r"^BENCHMARK_SLOW_LOG=(.*)$", re.M)
    b_error_re = re.compile(r"^BENCHMARK_ERROR_LOG=(.*)$", re.M)

    with open(benchmarking_ci_conf_path, "r") as configs_file:
        configs_str = configs_file.read()

        for m_variance in m_variance_re.findall(configs_str):
            metrics_variance = ast.literal_eval(m_variance.strip()[1:-1])
            if not isinstance(metrics_variance, dict):
                raise TypeError("Error in benchmark-ci.conf: METRICS_VARIANCE is not a python dict.")

        for m_recorded in m_recorded_re.findall(configs_str):
            metrics_recorded = ast.literal_eval(m_recorded.strip()[1:-1])
            if not isinstance(metrics_recorded, list):
                raise TypeError("Error in benchmark-ci.conf: METRICS_RECORDED is not a python list.")

        for perf_res in perf_res_re.findall(configs_str):
            PERF_RES_PATH = str(perf_res[1:-1])

        for output_path in output_path_re.findall(configs_str):
            OUTPUT_PATH = str(output_path[1:-1])

        for b_slow_log in b_slow_re.findall(configs_str):
            BENCHMARK_SLOW_LOG = str(b_slow_log[1:-1])

        for b_error_log in b_error_re.findall(configs_str):
            BENCHMARK_ERROR_LOG = str(b_error_log[1:-1])
        

def valid_timestamp(timestamp: str) -> bool:
    timestamp_re = re.compile(
        # YYYYMMDD_HHMMSS
        r"^\d{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])_(0[0-9]|1[0-9]|2[0-3])[0-5][0-9][0-5][0-9]$"
    )
    return timestamp_re.match(timestamp) is not None