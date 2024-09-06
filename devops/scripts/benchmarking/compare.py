import csv
import sys
from pathlib import Path

import common 

# TODO compare_to(metric) instead?
def compare_to_median(test_name: str, test_csv_path: str):
	median = dict()
	with open(f"{common.PERF_RES_PATH}/{test_name}/{test_name}-median.csv", mode='r') as median_csv:
		for stat in csv.DictReader(median_csv):
			median[stat["TestCase"]] = \
					{ metric: float(stat[metric]) for metric in common.metrics_variance }

	# TODO read status codes from a config file
	status = 0
	failure_counts = { metric: 0 for metric in common.metrics_variance }
	with open(test_csv_path, mode='r') as sample_csv:
		for sample in csv.DictReader(sample_csv):
			# Ignore test cases we haven't profiled before
			if sample["TestCase"] not in median:
				continue
			test_median = median[sample["TestCase"]]
			for metric, threshold in common.metrics_variance.items():
				max_tolerated = test_median[metric] * (1 + threshold)
				if common.sanitize(sample[metric]) >  max_tolerated:
					print("vvv FAILED vvv")
					print(sample['TestCase'])
					print(f"{metric}: {metric} {common.sanitize(sample[metric])} -- Historic avg. {test_median[metric]} (max tolerance {threshold*100}% -- {max_tolerated})")
					print("^^^^^^^^^^^^^^")
					status = 1
					failure_counts[metric] += 1
	if status != 0:
		print(f"Failure counts: {failure_counts}")
	return status


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print(f"Usage: {sys.argv[0]} <test name> <test csv path>")
		exit(-1)
	common.load_configs()
	exit(compare_to_median(sys.argv[1], sys.argv[2]))
