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
class DetectVersionsOptions:
    """
    Options for automatic version detection
    """

    # Components to detect versions for:
    sycl: bool = False
    compute_runtime: bool = False
    # umf: bool = False
    # level_zero: bool = False

    # Placeholder text, should automatic version detection fail: This text will
    # only be used if automatic version detection for x component is explicitly
    # specified.
    not_found_placeholder = "unknown"  # None

    # TODO unauthenticated users only get 60 API calls per hour: this will not
    # work if we enable benchmark CI in precommit.
    compute_runtime_tag_api: str = (
        "https://api.github.com/repos/intel/compute-runtime/tags"
    )
    # Max amount of api calls permitted on each run of the benchmark scripts
    max_api_calls = 4

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
    compute_runtime_tag: str = "25.22.33944.4"
    build_igc: bool = False
    current_run_name: str = "This PR"
    preset: str = "Full"
    build_jobs: int = multiprocessing.cpu_count()

    # Options intended for CI:
    regression_threshold: float = 0.05
    # It's necessary in CI to compare or redo benchmark runs. Instead of
    # generating a new timestamp each run by default, specify a single timestamp
    # to use across the entire CI run.
    timestamp_override: str = None
    # The default directory to fetch results from is args.benchmark_directory,
    # hence a default value of "None" as the value is decided during runtime.
    #
    # However, sometimes you may want to fetch results from a different
    # directory, i.e. in CI when you clone the results directory elsewhere.
    results_directory_override: str = None
    # By default, we fetch SYCL commit info from the folder where main.py is
    # located. This doesn't work right when CI uses different commits for e.g.
    # CI scripts vs SYCl build source.
    github_repo_override: str = None
    git_commit_override: str = None

    detect_versions: DetectVersionsOptions = field(
        default_factory=DetectVersionsOptions
    )


options = Options()
