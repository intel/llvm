# Copyright (C) 2025-2026 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
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

VERBOSE_LOGS = False

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

    def run_main(self, *args) -> subprocess.CompletedProcess:

        # TODO: not yet tested: "--detect-version", "sycl,compute_runtime"

        proc = subprocess.run(
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
                "--verbose" if VERBOSE_LOGS else "--log-level=info",
                *args,
            ],
            capture_output=True,
        )
        print(
            "MAIN_PY_STDOUT:",
            "\n" + proc.stdout.decode() if proc.stdout else " <empty>",
        )
        print(
            "MAIN_PY_STDERR:",
            "\n" + proc.stderr.decode() if proc.stderr else " <empty>",
        )
        return proc

    def get_benchmark_output_data(self):
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

    def _checkExistsInProcessOutput(
        self, proc: subprocess.CompletedProcess, expected: str
    ):
        """
        Check that expected regex string exists in process output.
        It's useful for checking e.g. if expected params are passed to the benchmark's bin execution.
        """
        stdout = proc.stdout.decode()
        self.assertRegex(stdout, expected, "Expected string not found in output")

    def _checkCase(
        self,
        caseName: str,
        groupName: str,
        tags: set[str],
        expected_in_output: str = None,
    ):
        return_proc = self.app.run_main("--filter", caseName + "$")
        self.assertEqual(return_proc.returncode, 0, "Subprocess did not exit cleanly")

        if expected_in_output:
            self._checkExistsInProcessOutput(return_proc, expected_in_output)

        out = self.app.get_benchmark_output_data()
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
            "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion",
            "SubmitKernel out of order with completion using events",
            {"L0", "latency", "micro", "submit"},
        )

    def test_torch_l0(self):
        self._checkCase(
            "torch_benchmark_l0 KernelSubmitSingleQueue kernelBatchSize 512, kernelDataType Int32, kernelName Add, kernelParamsNum 5, kernelSubmitPattern Single, kernelWGCount 4096, kernelWGSize 512",
            "KernelSubmitSingleQueue Int32Large",
            {"pytorch", "L0"},
            "--test=KernelSubmitSingleQueue.*--profilerType=timer",
        )
        self._checkCase(
            "torch_benchmark_l0 KernelSubmitSingleQueue kernelBatchSize 512, kernelDataType Int32, kernelName Add, kernelParamsNum 5, kernelSubmitPattern Single, kernelWGCount 4096, kernelWGSize 512 CPU count",
            "KernelSubmitSingleQueue Int32Large, CPU count",
            {"pytorch", "L0"},
            "--test=KernelSubmitSingleQueue.*--profilerType=cpuCounter",
        )
        self._checkCase(
            "torch_benchmark_l0 KernelSubmitMultiQueue kernelWGCount 4096, kernelWGSize 512, kernelsPerQueue 20, measureCompletion 0, useProfiling 0",
            "KernelSubmitMultiQueue large",
            {"pytorch", "L0"},
        )
        self._checkCase(
            "torch_benchmark_l0 KernelSubmitMultiQueue kernelWGCount 4096, kernelWGSize 512, kernelsPerQueue 20, measureCompletion 1, useProfiling 0 CPU count",
            "KernelSubmitMultiQueue large, CPU count",
            {"pytorch", "L0"},
        )
        self._checkCase(
            "torch_benchmark_l0 KernelSubmitSlmSize kernelBatchSize 512, measureCompletion 1, slmNum 1, useProfiling 0",
            "KernelSubmitSlmSize small",
            {"pytorch", "L0"},
        )
        self._checkCase(
            "torch_benchmark_l0 KernelSubmitLinearKernelSize kernelBatchSize 512, kernelSize 32",
            "KernelSubmitLinearKernelSize array32",
            {"pytorch", "L0"},
        )
        self._checkCase(
            "torch_benchmark_l0 KernelSubmitMemoryReuse kernelBatchSize 4096, kernelDataType Int32",
            "KernelSubmitMemoryReuse Int32Large",
            {"pytorch", "L0"},
        )
        self._checkCase(
            "torch_benchmark_l0 KernelSubmitGraphSingleQueue kernelBatchSize 512, kernelGroupsCount 32, kernelName Add, kernelWGCount 256, kernelWGSize 128, useProfiling 0",
            "KernelSubmitGraphSingleQueue small",
            {"pytorch", "L0"},
        )

    def test_torch_sycl(self):
        self._checkCase(
            "torch_benchmark_sycl KernelSubmitSingleQueue kernelBatchSize 512, kernelDataType Mixed, kernelName Add, kernelParamsNum 5, kernelSubmitPattern Single, kernelWGCount 512, kernelWGSize 256",
            "KernelSubmitSingleQueue MixedMedium",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_sycl KernelSubmitMultiQueue kernelWGCount 512, kernelWGSize 256, kernelsPerQueue 10, measureCompletion 1, useProfiling 0",
            "KernelSubmitMultiQueue medium",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_sycl KernelSubmitSlmSize kernelBatchSize 512, measureCompletion 0, slmNum 16384, useProfiling 0",
            "KernelSubmitSlmSize large",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_sycl KernelSubmitSlmSize kernelBatchSize 512, measureCompletion 1, slmNum 16384, useProfiling 0 CPU count",
            "KernelSubmitSlmSize large, CPU count",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_sycl KernelSubmitLinearKernelSize kernelBatchSize 512, kernelSize 5120",
            "KernelSubmitLinearKernelSize array5120",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_sycl KernelSubmitMemoryReuse kernelBatchSize 4096, kernelDataType Float",
            "KernelSubmitMemoryReuse FloatLarge",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_sycl KernelSubmitGraphSingleQueue kernelBatchSize 512, kernelGroupsCount 64, kernelName AddSequence, kernelWGCount 512, kernelWGSize 256, useProfiling 0",
            "KernelSubmitGraphSingleQueue medium",
            {"pytorch", "SYCL"},
        )

    def test_torch_syclpreview(self):
        self._checkCase(
            "torch_benchmark_syclpreview KernelSubmitSingleQueue kernelBatchSize 512, kernelDataType Mixed, kernelName Add, kernelParamsNum 5, kernelSubmitPattern Single, kernelWGCount 256, kernelWGSize 128",
            "KernelSubmitSingleQueue MixedSmall",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_syclpreview KernelSubmitMultiQueue kernelWGCount 256, kernelWGSize 128, kernelsPerQueue 4, measureCompletion 1, useProfiling 0",
            "KernelSubmitMultiQueue small",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_syclpreview KernelSubmitSlmSize kernelBatchSize 512, measureCompletion 1, slmNum 1024, useProfiling 0",
            "KernelSubmitSlmSize medium",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_syclpreview KernelSubmitLinearKernelSize kernelBatchSize 512, kernelSize 512",
            "KernelSubmitLinearKernelSize array512",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_syclpreview KernelSubmitLinearKernelSize kernelBatchSize 512, kernelSize 512 CPU count",
            "KernelSubmitLinearKernelSize array512, CPU count",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_syclpreview KernelSubmitMemoryReuse kernelBatchSize 512, kernelDataType Float",
            "KernelSubmitMemoryReuse FloatMedium",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_syclpreview KernelSubmitMemoryReuse kernelBatchSize 512, kernelDataType Float CPU count",
            "KernelSubmitMemoryReuse FloatMedium, CPU count",
            {"pytorch", "SYCL"},
        )
        self._checkCase(
            "torch_benchmark_syclpreview KernelSubmitGraphSingleQueue kernelBatchSize 512, kernelGroupsCount 128, kernelName Empty, kernelWGCount 4096, kernelWGSize 512, useProfiling 0",
            "KernelSubmitGraphSingleQueue large",
            {"pytorch", "SYCL"},
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SYCL's benchmark test framework")
    parser.add_argument(
        "--verbose",
        help="Set benchmark framework's logging level to DEBUG.",
        action="store_true",
    )

    args = parser.parse_args()
    VERBOSE_LOGS = args.verbose

    unittest.main()
