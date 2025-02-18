from dataclasses import dataclass, field
from enum import Enum


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
    benchmark_cwd: str = "INVALID"
    timeout: float = 600
    iterations: int = 3
    verbose: bool = False
    compare: Compare = Compare.LATEST
    compare_max: int = 10  # average/median over how many results
    output_markdown: MarkdownSize = MarkdownSize.SHORT
    output_html: bool = False
    dry_run: bool = False
    # these two should probably be merged into one setting
    stddev_threshold: float = 0.02
    epsilon: float = 0.02
    iterations_stddev: int = 5
    build_compute_runtime: bool = False
    extra_ld_libraries: list[str] = field(default_factory=list)
    extra_env_vars: dict = field(default_factory=dict)
    compute_runtime_tag: str = "24.52.32224.10"
    build_igc: bool = False
    current_run_name: str = "This PR"


options = Options()
