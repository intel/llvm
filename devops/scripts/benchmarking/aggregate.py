import csv
import sys
from pathlib import Path
import heapq
import statistics

import common


# Simple median calculation
class SimpleMedian:

    def __init__(self):
        self.elements = []

    def add(self, n: float):
        self.elements.append(n)

    def get_median(self) -> float:
        return statistics.median(self.elements)


# Calculate medians incrementally using a heap: Useful for when dealing with
# large number of samples.
#
# TODO how many samples are we going to realistically get? I had written this
# with precommit in mind, but if this only runs nightly, it would actually be
# faster to do a normal median calculation.
class StreamingMedian:

    def __init__(self):
        # Gist: we keep a minheap and a maxheap, and store the median as the top
        # of the minheap. When a new element comes it gets put into the heap
        # based on if the element is bigger than the current median. Then, the
        # heaps are heapified and the median is repopulated by heapify.
        self.minheap_larger = []
        self.maxheap_smaller = []

    # Note: numbers on maxheap should be negative, as heapq
    # is minheap by default

    def add(self, n: float):
        if len(self.maxheap_smaller) == 0 or -self.maxheap_smaller[0] >= n:
            heapq.heappush(self.maxheap_smaller, -n)
        else:
            heapq.heappush(self.minheap_larger, n)

        # Ensure minheap has more elements than maxheap
        if len(self.maxheap_smaller) > len(self.minheap_larger) + 1:
            heapq.heappush(self.minheap_larger, -heapq.heappop(self.maxheap_smaller))
        elif len(self.maxheap_smaller) < len(self.minheap_larger):
            heapq.heappush(self.maxheap_smaller, -heapq.heappop(self.minheap_larger))

    def get_median(self) -> float:
        if len(self.maxheap_smaller) == len(self.minheap_larger):
            # Equal number of elements smaller and larger than "median":
            # thus, there are two median values. The median would then become
            # the average of both median values.
            return (-self.maxheap_smaller[0] + self.minheap_larger[0]) / 2.0
        else:
            # Otherwise, median is always in minheap, as minheap is always
            # bigger
            return -self.maxheap_smaller[0]


def aggregate_median(test_name: str, test_dir: str, cutoff: str):

    # Get all .csv samples for the requested test folder
    def csv_samples() -> list[str]:
        # TODO check that the path below is valid directory
        cache_dir = Path(f"{test_dir}")
        # TODO check for time range; What time range do I want?
        return filter(
            lambda f: f.is_file()
            and common.valid_timestamp(str(f)[-19:-4])
            and str(f)[-19:-4] > cutoff,
            cache_dir.glob(f"{test_name}-*_*.csv"),
        )

    # Calculate median of every desired metric:
    aggregate_s = dict()
    for sample_path in csv_samples():
        with open(sample_path, "r") as sample_file:
            for s in csv.DictReader(sample_file):
                test_case = s["TestCase"]
                # Construct entry in aggregate_s for test case if it does not
                # exist already:
                if test_case not in aggregate_s:
                    aggregate_s[test_case] = {
                        metric: SimpleMedian() for metric in common.metrics_variance
                    }

                for metric in common.metrics_variance:
                    aggregate_s[test_case][metric].add(common.sanitize(s[metric]))

    # Write calculated median (aggregate_s) as a new .csv file:
    with open(
        f"{test_dir}/{test_name}-median.csv", "w"
    ) as output_csv:
        writer = csv.DictWriter(
            output_csv, fieldnames=["TestCase", *common.metrics_variance.keys()]
        )
        writer.writeheader()
        for test_case in aggregate_s:
            writer.writerow(
                {"TestCase": test_case}
                | {
                    metric: aggregate_s[test_case][metric].get_median()
                    for metric in common.metrics_variance
                }
            )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <test name> <absolute path to test directory> <cutoff timestamp YYYYMMDD_HHMMSS>"
        )
        exit(1)
    if not common.valid_timestamp(sys.argv[3]):
        print(sys.argv)
        print(f"Bad cutoff timestamp, please use YYYYMMDD_HHMMSS.")
        exit(1)
    common.load_configs()

    aggregate_median(sys.argv[1], sys.argv[2], sys.argv[3])
