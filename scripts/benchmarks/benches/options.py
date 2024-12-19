from dataclasses import dataclass
from enum import Enum

class Compare(Enum):
    LATEST = 'latest'
    AVERAGE = 'average'
    MEDIAN = 'median'

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
    compare_max: int = 10 # average/median over how many results
    output_html: bool = False
    output_markdown: bool = True
    dry_run: bool = False
    # these two should probably be merged into one setting
    stddev_threshold: float = 0.02
    epsilon: float = 0.02

options = Options()

