import csv
import sys
from pathlib import Path
import heapq

import common

class StreamingMedian:
	
    def __init__(self):
        self.minheap_larger = []
        self.maxheap_smaller = []
		# Note: numbers on maxheap should be negative, as heapq
        # is minheap by default

    def add(self, n: float):
        if len(self.maxheap_smaller) == 0 or -self.maxheap_smaller[0] >= n:
            heapq.heappush(self.maxheap_smaller, -n)
        else:
            heapq.heappush(self.minheap_larger, n)

        if len(self.maxheap_smaller) > len(self.minheap_larger) + 1:
            heapq.heappush(self.minheap_larger,
						   -heapq.heappop(self.maxheap_smaller))
        elif len(self.maxheap_smaller) < len(self.minheap_larger):
            heapq.heappush(self.maxheap_smaller,
						   -heapq.heappop(self.minheap_larger))

    def get_median(self) -> float:
        if len(self.maxheap_smaller) == len(self.minheap_larger):
            return (-self.maxheap_smaller[0] + self.minheap_larger[0]) / 2.0
        else:
            return -self.maxheap_smaller[0]


def aggregate_median(benchmark: str, cutoff: str):

	def csv_samples() -> list[str]:
		# TODO check that the path below is valid directory
		with Path(f"{common.PERF_RES_PATH}/{benchmark}") as cache_dir:
			# TODO check for time range; What time range do I want?
			return filter(lambda f: f.is_file() and
						  common.valid_timestamp(str(f)[-13:]) and str(f)[-13:] > cutoff,
						  cache_dir.glob(f"{benchmark}-*_*.csv"))
	
	# Calculate median of every desired metric:
	aggregate_s = dict()
	for sample_path in csv_samples():
		with open(sample_path, mode='r') as sample_file:
			for s in csv.DictReader(sample_file):
				if s["TestCase"] not in aggregate_s:
					aggregate_s[s["TestCase"]] = \
				 		{ metric: StreamingMedian() for metric in common.metrics_variance }
				for metric in common.metrics_variance:
					aggregate_s[s["TestCase"]][metric].add(common.sanitize(s[metric]))

	with open(f"{common.PERF_RES_PATH}/{benchmark}/{benchmark}-median.csv", 'w') as output_csv:
		writer = csv.DictWriter(output_csv,
							    fieldnames=["TestCase", *common.metrics_variance.keys()])
		writer.writeheader()
		for test_case in aggregate_s:
			writer.writerow({ "TestCase": test_case } | 
				{ metric: aggregate_s[test_case][metric].get_median() 
					for metric in common.metrics_variance })
	
		
if __name__ == "__main__":
	if len(sys.argv) < 3:
		print(f"Usage: {sys.argv[0]} <test case name> <cutoff date YYMMDD_HHMMSS>")
		exit(1)
	if not common.valid_timestamp(sys.argv[2]):
		print(f"Bad cutoff timestamp, please use YYMMDD_HHMMSS.")
		exit(1)

	aggregate_median(sys.argv[1], sys.argv[2])
