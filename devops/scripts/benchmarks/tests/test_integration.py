# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import unittest
import tempfile
import subprocess
import json
from collections import namedtuple

# oneapi has to be installed and sourced for sycl benchmarks tests

DataJson = namedtuple("DataJson", ["runs", "metadata", "tags", "names"])
DataJsonRun = namedtuple("DataJsonRun", ["name", "results"])
DataJsonResult = namedtuple(
    "DataJsonResult", ["name", "label", "suite", "value", "unit"]
)

class App:
    def __init__(self):
        self.OUTPUT_DIR = None
        self.RESULTS_DIR = None
        self.WORKDIR_DIR = None

    def prepare_dirs(self):
        self.OUTPUT_DIR = tempfile.mkdtemp()
        self.RESULTS_DIR = tempfile.mkdtemp()
        self.WORKDIR_DIR = tempfile.mkdtemp()

        # when UT does not want to build compute-benchmarks from scratch, it can provide prebuilt path
        cb_targetpath = os.environ.get("COMPUTE_BENCHMARKS_BUILD_PATH")
        if cb_targetpath and os.path.isdir(cb_targetpath):
            cb_build_dir = os.path.join(self.WORKDIR_DIR, "compute-benchmarks-build")
            os.symlink(cb_targetpath, cb_build_dir)
            with open(
                os.path.join(self.WORKDIR_DIR, "BENCH_WORKDIR_VERSION"), "w"
            ) as f:
                f.write("2.0")  # TODO: take from main.INTERNAL_WORKDIR_VERSION

    def remove_dirs(self):
        for d in [self.RESULTS_DIR, self.OUTPUT_DIR, self.WORKDIR_DIR]:
            if d is not None:
                shutil.rmtree(d, ignore_errors=True)

    def run_main(self, *args):

        # TODO: not yet tested: "--detect-version", "sycl,compute_runtime"

        return subprocess.run(
            [
                "./devops/scripts/benchmarks/main.py",
                self.WORKDIR_DIR,
                "--sycl",
                os.environ.get("CMPLR_ROOT"),
                "--save",
                "testfile",
                "--output-html",
                "remote",
                "--results-dir",
                self.RESULTS_DIR,
                "--output-dir",
                self.OUTPUT_DIR,
                "--preset",
                "Minimal",
                "--timestamp-override",
                "20240102_030405",
                "--stddev-threshold",
                "999999999.9",
                "--exit-on-failure",
                *args,
            ]
        )

    def get_output(self):
        with open(os.path.join(self.OUTPUT_DIR, "data.json")) as f:
            out = json.load(f)
            return DataJson(
                runs=[
                    DataJsonRun(
                        name=run["name"],
                        results=[
                            DataJsonResult(
                                name=r["name"],
                                label=r["label"],
                                suite=r["suite"],
                                value=r["value"],
                                unit=r["unit"],
                            )
                            for r in run["results"]
                        ],
                    )
                    for run in out["benchmarkRuns"]
                ],
                metadata=out["benchmarkMetadata"],
                tags=out["benchmarkTags"],
                names=out["defaultCompareNames"],
            )


# add "--verbose" for debug logs


class TestE2E(unittest.TestCase):
    def setUp(self):
        # Load test data
        self.app = App()
        self.app.remove_dirs()
        self.app.prepare_dirs()

        # clean directory with input, output

    def tearDown(self):
        self.app.remove_dirs()

    def test_record_and_replay(self):
        caseName = "L0 RecordGraph AppendCopy 1, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 2, Instantiations 10, Lvls 4, Rec"
        run_result = self.app.run_main("--filter", caseName + "$")
        self.assertEqual(run_result.returncode, 0, "Subprocess did not exit cleanly")

        out = self.app.get_output()

        self.assertIn(caseName, [r.name for r in out.runs[0].results])

        metadata = out.metadata[caseName]
        self.assertEqual(metadata["type"], "benchmark")
        self.assertEqual(set(metadata["tags"]), {"L0"})

    def test_submit_kernel(self):
        caseName = "SubmitKernel out of order with measure completion KernelExecTime=20"
        run_result = self.app.run_main("--filter", caseName + "$")
        self.assertEqual(run_result.returncode, 0, "Subprocess did not exit cleanly")

        out = self.app.get_output()

        testName = "api_overhead_benchmark_l0 " + caseName
        self.assertIn(testName, [r.name for r in out.runs[0].results])

        metadata = out.metadata[testName]
        self.assertEqual(metadata["type"], "benchmark")
        self.assertEqual(set(metadata["tags"]), {"L0", "latency", "micro", "submit"})


if __name__ == "__main__":
    unittest.main()
