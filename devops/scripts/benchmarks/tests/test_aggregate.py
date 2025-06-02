import sys
import os

sys.path.append(f"{os.path.dirname(__file__)}/../")
from options import options
from utils.aggregate import *

def run_testcase(aggregator: Aggregator, src: list, expected: float) -> bool:
    aggr = aggregator()
    for n in src:
        aggr.add(n)
    res = aggr.get_avg()
    if res != expected:
        print(f"Failed: {aggregator}, {src} -- expected {expected}, got {res}")
        return False
    return True


def test_EWMA():
    options.EWMA_smoothing_factor = 0.5
    testcases = [
        ([], None),
        ([100], 100),
        ([100, 100, 100, 100, 100], 100),
        ([100, 105, 103, 108, 107], 106.1875),
    ]
    successes = 0
    fails = 0
    for t in testcases:
        if not run_testcase(EWMA, *t):
            fails = fails + 1
        else:
            successes = successes + 1
    print(f"EWMA test: {successes} successes, {fails} fails.")


if __name__ == "__main__":
    test_EWMA()

