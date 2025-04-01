from dataclasses import dataclass, field
from enum import Enum
import multiprocessing

from presets import presets


class Compare(Enum):
    LATEST = "latest"
    AVERAGE = "average"
    MEDIAN = "median"


class MarkdownSize(Enum):
    SHORT = "short"
    FULL = "full"


@dataclass
class Options:
    workdir: str = None
    sycl: str = None
    ur: str = None
    ur_adapter: str = None
    umf: str = None
    rebuild: bool = True
    redownload: bool = False
    benchmark_cwd: str = "INVALID"
    timeout: float = 600
    iterations: int = 3
    verbose: bool = False
    compare: Compare = Compare.LATEST
    compare_max: int = 10  # average/median over how many results
    output_markdown: MarkdownSize = MarkdownSize.SHORT
    output_html: str = "local"
    output_directory: str = None
    dry_run: bool = False
    stddev_threshold: float = 0.02
    iterations_stddev: int = 5
    build_compute_runtime: bool = False
    extra_ld_libraries: list[str] = field(default_factory=list)
    extra_env_vars: dict = field(default_factory=dict)
    compute_runtime_tag: str = "25.05.32567.18"
    build_igc: bool = False
    current_run_name: str = "This PR"
    preset: str = "Full"
    build_jobs: int = multiprocessing.cpu_count()

    # Options applicable to CI only:
    regression_threshold: float = 0.05
    # In CI, it may be necessary to e.g. compare or redo benchmark runs.
    # A timestamp is generated at the beginning of the CI run and used through
    # the entire CI process, instead of scripts generating their own timestamps
    # every time a script runs (default behavior).
    timestamp_override: str = None
    # By default, the directory to fetch results from is the benchmark working
    # directory specified in the CLI args, hence a default value of "None" as
    # the value is decided via runtime.
    #
    # However, sometimes you may want to fetch results from a different
    # directory, i.e. in CI when you clone the results directory elsewhere.
    results_directory_override: str = None


options = Options()
