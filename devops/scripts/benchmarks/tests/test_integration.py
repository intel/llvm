# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import namedtuple

sys.path.append(f"{os.path.dirname(__file__)}/../")
from utils.workdir_version import INTERNAL_WORKDIR_VERSION


DataJson = namedtuple("DataJson", ["runs", "metadata", "tags", "names"])
DataJsonRun = namedtuple("DataJsonRun", ["name", "results"])
DataJsonResult = namedtuple(
    "DataJsonResult", ["name", "label", "suite", "value", "unit"]
)
DataJsonMetatdata = namedtuple(
    "DataJsonMetatdata",
    [
        "type",
        "unstable",
        "tags",
        "range_min",
        "range_max",
        "display_name",
        "explicit_group",
    ],
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
                f.write(INTERNAL_WORKDIR_VERSION)

    def remove_dirs(self):
        for d in [self.RESULTS_DIR, self.OUTPUT_DIR, self.WORKDIR_DIR]:
            if d is not None:
                shutil.rmtree(d, ignore_errors=True)

    def run_main(self, *args):

        # TODO: not yet tested: "--detect-version", "sycl,compute_runtime"

        procesResult = subprocess.run(
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
            ],
            capture_output=True,
        )
        print("MAIN_PY_STDOUT:\n" + procesResult.stdout.decode())
        print("MAIN_PY_STDERR:\n" + procesResult.stderr.decode())
        return procesResult.returncode

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
                metadata=dict(
                    [
                        (
                            k,
                            DataJsonMetatdata(
                                type=v["type"],
                                unstable=v.get("unstable", False),
                                tags=v.get("tags", []),
                                range_min=v.get("range_min"),
                                range_max=v.get("range_max"),
                                display_name=v.get("display_name"),
                                explicit_group=v.get("explicit_group"),
                            ),
                        )
                        for k, v in out["benchmarkMetadata"].items()
                    ]
                ),
                tags=out["benchmarkTags"],
                names=out["defaultCompareNames"],
            )


# add "--verbose" for debug logs


class TestE2E(unittest.TestCase):
    def setUp(self):
        # Load test data
        print(f"::group::{self._testMethodName}")
        self.app = App()
        self.app.remove_dirs()
        self.app.prepare_dirs()

    def tearDown(self):
        self.app.remove_dirs()
        print(f"::endgroup::")

    def _checkGroup(
        self, expectedGroupName: str, benchMetadata: DataJsonMetatdata, out: DataJson
    ):
        self.assertEqual(benchMetadata.type, "benchmark")
        benchmarkGroupName = benchMetadata.explicit_group
        self.assertEqual(benchmarkGroupName, expectedGroupName)
        groupMetadata = out.metadata[benchmarkGroupName]
        self.assertEqual(groupMetadata.type, "group")

    def _checkResultsExist(self, caseName: str, out: DataJson):
        self.assertIn(caseName, [r.name for r in out.runs[0].results])

    def _checkCase(self, caseName: str, groupName: str, tags: set[str]):
        run_result = self.app.run_main("--filter", caseName + "$")
        self.assertEqual(run_result, 0, "Subprocess did not exit cleanly")

        out = self.app.get_output()
        self._checkResultsExist(caseName, out)

        metadata = out.metadata[caseName]
        self.assertEqual(set(metadata.tags), tags)
        self._checkGroup(groupName, metadata, out)

    def test_record_and_replay(self):
        self._checkCase(
            "record_and_replay_benchmark_l0 AppendCopy 1, AppendKern 10, CmdSetsInLvl 10, ForksInLvl 2, Instantiations 10, Lvls 4, Rec",
            "RecordGraph large",
            {"L0"},
        )

    def test_submit_kernel(self):
        self._checkCase(
            "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion KernelExecTime=20",
            "SubmitKernel out of order with completion using events long kernel",
            {"L0", "latency", "micro", "submit"},
        )


if __name__ == "__main__":
    unittest.main()
