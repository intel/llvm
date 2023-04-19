# -*- Python -*-
"""Tool parses output of FloydWarshall programs and compares relatively
the execution time both of them.
"""

import argparse


def read_exec_result(filename):
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith("Time: "):
                continue
            
            line = line.strip("Time: ")
            return float(line.split()[0])
        
        raise Exception('No found the following line: "Time: ", filename: {0}'.format(filename))


parser = argparse.ArgumentParser(prog="compare.py")
parser.add_argument("--files", action="store", dest="files",
                    help="2 files that are expected to be outputs of FloydWarshall program runs",
                    required=True)
parser.add_argument("--diff", action="store", dest="diff",
                    help="Relative diff used to compare two execution times",
                    type=float, required=True)

args = parser.parse_args()
files = args.files
files = files.split(',')
if len(files) != 2:
    raise Exception("Only 2 output files are expected")

diff = args.diff
if not (0. < diff < 1.):
    raise "relative diff must be in (0, 1)"

exec_results = [read_exec_result(f) for f in files]
if exec_results[1] > diff * exec_results[0]:
    print("Test failed: Exec times: {0}, diff: {1}".format(exec_results, diff))
    exit(1)

print("OK")
