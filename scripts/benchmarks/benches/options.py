from dataclasses import dataclass

@dataclass
class Options:
    sycl: str = ""
    rebuild: bool = True
    benchmark_cwd: str = "INVALID"
    timeout: float = 600
    iterations: int = 5
    verbose: bool = False

options = Options()

