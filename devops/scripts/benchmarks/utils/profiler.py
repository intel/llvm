from abc import ABC
from tempfile import mktemp

class Profiler(ABC):
    """
    Represents a performance analysis tool and the information needed to:
    - Run benchmarks using the performance analysis tool,
    - Collect information produced by said tool
    """ 

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def command_prefix(self, *args, **kwargs) -> str:
        """
        CLI prefix to append to benchmark command, in order to run the benchmark
        using the given performance analyzer.
        
        The intention is to output performance data into a file (in e.g. /tmp),
        and then read the data in said file later via parse_output: Many
        benchmarks parse stdout to obtain benchmark results, thus polluting
        stdout will cause problems when collecting benchmark data itself.
        """
        pass

    @abstractmethod
    def parse_output(self, *args, **kwargs) -> dict:
        """
        Parses output of performance analyzer and returns a dict containing
        performance information parsed from said output.
        """
        pass

class Perf(Profiler):
    """
    Run benchmarks using `perf stat`
    """

    def __init__(self, events: list[str] = ["instructions"]):
        self.output_file = mktemp()
        self.events = events

    def name(self) -> str:
        return "perf"

    def command_prefix(self) -> str:
        return f"perf stat -e {",".join(self.events)} -o {self.output_file} "

    def parse_output(self) -> dict:
        return {
            
        }
        