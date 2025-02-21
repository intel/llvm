import re
import os
import sys
import string
import configparser


class Validate:
    """Static class containing methods for validating various fields"""

    @staticmethod
    def filepath(path: str) -> bool:
        """
        Returns True if path is clean (no illegal characters), otherwise False.
        """
        filepath_re = re.compile(r"[a-zA-Z0-9\/\._\-]+")
        return filepath_re.match(path) is not None

    @staticmethod
    def timestamp(t: str) -> bool:
        """
        Returns True if t is in form YYYYMMDD_HHMMSS, otherwise False.
        """
        timestamp_re = re.compile(
            r"^\d{4}(0[1-9]|1[0-2])([0-2][0-9]|3[01])_([01][0-9]|2[0-3])[0-5][0-9][0-5][0-9]$"
        )
        return timestamp_re.match(t) is not None

    @staticmethod
    def sanitize_stat(stat: str) -> float:
        """
        Sanitize statistics found in compute-benchmark output csv files. Returns
        float if sanitized, None if not sanitizable.
        """
        # Get rid of %
        if stat[-1] == "%":
            stat = stat[:-1]

        # Cast to float: If cast succeeds, the statistic is clean.
        try:
            return float(stat)
        except ValueError:
            return None


class SanitizedConfig:
    """
    Static class for holding sanitized configuration values used within python.

    Configuration option names follow <section name>_<option name> from config
    file.
    """

    loaded: bool = False
    # PERF_RES_PATH: str = None
    # ARTIFACT_OUTPUT_CACHE: str = None
    METRICS_TOLERANCES: dict = None
    METRICS_RECORDED: list = None
    # BENCHMARK_LOG_SLOW: str = None
    # BENCHMARK_LOG_ERROR: str = None

    @staticmethod
    def load(devops_path: str):
        config = Configuration(devops_path)
        config.export_python_globals()


class Configuration:
    """
    Class handling loading, sanitizing, and exporting configuration options for
    use within python or shell scripts.
    """

    def __init__(self, devops_path: str):
        """
        Initialize this configuration handler by finding configuration files

        @param devops_path Path to /devops folder in intel/llvm
        """
        self.config_path = f"{devops_path}/benchmarking/config.ini"
        self.constants_path = f"{devops_path}/benchmarking/constants.ini"

        if not os.path.isfile(self.config_path):
            print(
                f"config.ini not found in {devops_path}/benchmarking.", file=sys.stderr
            )
            exit(1)
        if not os.path.isfile(self.constants_path):
            print(
                f"constants.ini not found in {devops_path}/benchmarking.",
                file=sys.stderr,
            )
            exit(1)

    def __sanitize(self, value: str, field: str) -> str:
        """
        Enforces an allowlist of characters and sanitizes input from config
        files.
        """
        _alnum = list(string.ascii_letters + string.digits)
        allowlist = _alnum + ["_", "-", ".", ",", ":", "/", "%"]

        for illegal_ch in filter(lambda ch: ch not in allowlist, value):
            print(f"Illegal character '{illegal_ch}' in {field}", file=sys.stderr)
            exit(1)

        return value

    def __get_export_cmd(self, export_opts: list, config_file_path: str) -> str:
        """
        Generates export commands for variables in the configuration file at
        config_file_path, as listed by export_opts.

        export_opts is list of tuples in (<option section>, <option name>) form.
        """
        config = configparser.ConfigParser()
        config.read(config_file_path)

        def export_var_cmd(sec: str, opt: str) -> str:
            var_name = f"SANITIZED_{sec.upper()}_{opt.upper()}"
            var_val = f"{self.__sanitize(config[sec][opt], sec + '.' + opt)}"
            return f"{var_name}={var_val}"

        export_cmds = [export_var_cmd(sec, opt) for sec, opt in export_opts]
        return "export " + " ".join(export_cmds)

    def export_shell_configs(self) -> str:
        """
        Return shell command exporting environment variables representing
        various configuration options used in shell scripts.
        """
        # List of configs used in shell scripts: Export only what's needed
        shell_configs = [
            ("compute_bench", "compile_jobs"),
            ("compute_bench", "iterations"),
            ("average", "cutoff_range"),
            ("average", "min_threshold"),
            ("device_selector", "enabled_backends"),
            ("device_selector", "enabled_devices"),
        ]
        return self.__get_export_cmd(shell_configs, self.config_path)

    def export_shell_constants(self) -> str:
        """
        Return shell command exporting environment variables representing
        various constants used in shell scripts.
        """
        # List of configs used in shell scripts: Export only what's needed
        shell_constants = [
            ("perf_res", "git_repo"),
            ("perf_res", "git_branch"),
            ("compute_bench", "git_repo"),
            ("compute_bench", "git_branch"),
            ("compute_bench", "git_commit"),
        ]
        return self.__get_export_cmd(shell_constants, self.constants_path)

    def export_python_globals(self):
        """
        Populate all configs/constants used in python into SanitizedConfig.
        """
        all_opts = configparser.ConfigParser()
        all_opts.read(self.config_path)
        all_opts.read(self.constants_path)

        # Fields that are supposed to be python objects need to be changed to
        # python objects manually:

        # metrics.recorded
        m_rec_str = self.__sanitize(all_opts["metrics"]["recorded"], "metrics.recorded")
        SanitizedConfig.METRICS_RECORDED = m_rec_str.split(",")

        # metrics.tolerances
        m_tol_str = self.__sanitize(
            all_opts["metrics"]["tolerances"], "metrics.tolerances"
        )
        metric_tolerances = dict(
            [pair_str.split(":") for pair_str in m_tol_str.split(",")]
        )

        for metric, tolerance_str in metric_tolerances.items():
            if metric not in SanitizedConfig.METRICS_RECORDED:
                print(
                    f"Metric compared against {metric} is not being recorded.",
                    file=sys.stderr,
                )
                exit(1)
            try:
                metric_tolerances[metric] = float(tolerance_str)
            except ValueError:
                print(f"Could not convert '{tolerance_str}' to float.", file=sys.stderr)
                exit(1)

        SanitizedConfig.METRICS_TOLERANCES = metric_tolerances

        SanitizedConfig.loaded = True
