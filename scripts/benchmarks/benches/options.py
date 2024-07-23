from dataclasses import dataclass

@dataclass
class Options:
    sycl: str = ""
    rebuild: bool = True
    benchmark_cwd: str = "INVALID"

options = Options()

