#!/usr/bin/env python3
"""Times a compiler command and prints PERF_RESULT lines.

Usage:
  perf_compile.py --benchmark-name "<name>" [--reps N] [--warmup N] -- <command...>

The command is run warmup + reps times. For each timed repetition the wall
time and the CPU time (user + sys of the whole child process tree) are
recorded. Two lines are printed, "<name>" (wall) and "<name>, CPU time", in
the format described in README.md. Exits nonzero if any run fails.
"""

import argparse
import json
import re
import resource
import statistics
import subprocess
import sys
import time

LABEL_RE = re.compile(r"[A-Za-z0-9 _.,:;<>()=+/*&#-]{1,120}")


def child_cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def result_line(name, values):
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    fields = [
        ("name", json.dumps(name)),
        ("variant", '""'),
        ("unit", '"s"'),
        ("value", "%.9g" % statistics.median(values)),
        ("stddev", "%.9g" % stddev),
        ("min", "%.9g" % min(values)),
        ("reps", str(len(values))),
        ("lower_is_better", "true"),
    ]
    return "PERF_RESULT {" + ",".join('"%s":%s' % f for f in fields) + "}"


def main(argv):
    if "--" not in argv:
        sys.exit("perf_compile.py: error: missing '--' before the command")
    split = argv.index("--")
    command = argv[split + 1 :]

    parser = argparse.ArgumentParser(prog="perf_compile.py")
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args(argv[:split])

    if not LABEL_RE.fullmatch(args.benchmark_name + ", CPU time"):
        sys.exit(
            "perf_compile.py: error: --benchmark-name %r is empty, too long or contains "
            "characters outside [A-Za-z0-9 _.,:;<>()=+/*&#-]" % args.benchmark_name
        )
    if args.reps < 1 or args.warmup < 0:
        sys.exit("perf_compile.py: error: need --reps >= 1 and --warmup >= 0")
    if not command:
        sys.exit("perf_compile.py: error: empty command")

    wall, cpu = [], []
    for i in range(args.warmup + args.reps):
        cpu_start = child_cpu_seconds()
        start = time.perf_counter_ns()
        proc = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        elapsed = (time.perf_counter_ns() - start) / 1e9
        cpu_elapsed = child_cpu_seconds() - cpu_start
        if proc.returncode != 0 or (i == 0 and proc.stdout):
            sys.stderr.write(proc.stdout)
        if proc.returncode != 0:
            sys.exit(
                "perf_compile.py: error: command exited with %d: %s"
                % (proc.returncode, " ".join(command))
            )
        if i >= args.warmup:
            wall.append(elapsed)
            cpu.append(cpu_elapsed)

    print(result_line(args.benchmark_name, wall))
    print(result_line(args.benchmark_name + ", CPU time", cpu))
    sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
