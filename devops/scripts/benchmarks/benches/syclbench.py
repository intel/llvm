# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import csv
from pathlib import Path

from .base import Benchmark, Suite, TracingType
from utils.result import Result
from options import options
from git_project import GitProject


class SyclBench(Suite):
    def __init__(self):
        self.project = None

    def name(self) -> str:
        return "SYCL-Bench"

    def git_url(self) -> str:
        return "https://github.com/unisa-hpc/sycl-bench.git"

    def git_hash(self) -> str:
        return "31fc70be6266193c4ba60eb1fe3ce26edee4ca5b"

    def setup(self) -> None:
        if options.sycl is None:
            return

        if self.project is None:
            self.project = GitProject(
                self.git_url(),
                self.git_hash(),
                Path(options.workdir),
                "sycl-bench",
                use_installdir=False,
            )

        if not self.project.needs_rebuild():
            log.info(f"Rebuilding {self.project.name} skipped")
            return

        extra_args = [
            f"-DCMAKE_CXX_COMPILER={options.sycl}/bin/clang++",
            f"-DCMAKE_C_COMPILER={options.sycl}/bin/clang",
            f"-DSYCL_IMPL=dpcpp",
        ]
        if options.ur_adapter == "cuda":
            extra_args += [
                f"-DCMAKE_CXX_FLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda"
            ]
        if options.ur_adapter == "hip":
            extra_args += [
                f"-DCMAKE_CXX_FLAGS=-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch={options.hip_arch}"
            ]

        self.project.configure(extra_args, add_sycl=True)
        self.project.build(add_sycl=True)

    def benchmarks(self) -> list[Benchmark]:
        return [
            # Blocked_transform(self), # run time < 1ms
            DagTaskI(self),
            DagTaskS(self),
            HostDevBandwidth(self),
            LocalMem(self),
            # Pattern_L2(self), # validation failure
            # Reduction(self), # validation failure
            ScalarProd(self),
            SegmentReduction(self),
            # UsmAccLatency(self), # validation failure
            UsmAllocLatency(self),
            # UsmInstrMix(self), # validation failure
            # UsmPinnedOverhead(self), # validation failure
            VecAdd(self),
            # *** sycl-bench single benchmarks
            # TwoDConvolution(self), # run time < 1ms
            Two_mm(self),
            Three_mm(self),
            # Arith(self), # run time < 1ms
            Atax(self),
            # Atomic_reduction(self), # run time < 1ms
            Bicg(self),
            # Correlation(self), # validation failure
            # Covariance(self), # validation failure
            # Gemm(self), # validation failure
            # Gesumv(self), # validation failure
            # Gramschmidt(self), # validation failure
            KMeans(self),
            # LinRegCoeff(self), # FIXME: causes serious GPU hangs on 25.31.34666.3
            # LinRegError(self), # run time < 1ms
            # MatmulChain(self), # validation failure
            MolDyn(self),
            # Mvt(self), # validation failure
            Sf(self),
            # Syr2k(self), # validation failure
            # Syrk(self), # validation failure
        ]


class SyclBenchmark(Benchmark):
    def __init__(self, suite: SyclBench, name: str, test: str):
        super().__init__(suite)
        self.suite = suite
        self.bench_name = name
        self.test = test

    @property
    def benchmark_bin(self) -> Path:
        return self.suite.project.build_dir / self.bench_name

    def enabled(self) -> bool:
        return options.sycl is not None

    def bin_args(self) -> list[str]:
        return []

    def extra_env_vars(self) -> dict:
        return {}

    def get_tags(self):
        base_tags = ["SYCL", "micro"]
        if "Memory" in self.bench_name or "mem" in self.bench_name.lower():
            base_tags.append("memory")
        if "Reduction" in self.bench_name:
            base_tags.append("math")
        if "Bandwidth" in self.bench_name:
            base_tags.append("throughput")
        if "Latency" in self.bench_name:
            base_tags.append("latency")
        return base_tags

    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        self.outputfile = os.path.join(options.workdir, self.test + ".csv")

        command = [
            str(self.benchmark_bin),
            f"--warmup-run",
            f"--num-runs={options.iterations}",
            f"--output={self.outputfile}",
        ]

        command += self.bin_args()
        env_vars.update(self.extra_env_vars())

        # no output to stdout, all in outputfile
        self.run_bench(command, env_vars, run_trace=run_trace, force_trace=force_trace)

        with open(self.outputfile, "r") as f:
            reader = csv.reader(f)
            res_list = []
            for row in reader:
                if not row[0].startswith("#"):
                    # Check if the test passed
                    if row[1] != "PASS":
                        raise Exception(f"{row[0]} failed")
                    res_list.append(
                        Result(
                            label=f"{self.name()} {row[0]}",
                            value=float(row[12]) * 1000,  # convert to ms
                            command=command,
                            env=env_vars,
                            unit="ms",
                            git_url=self.suite.git_url(),
                            git_hash=self.suite.git_hash(),
                        )
                    )

        os.remove(self.outputfile)

        return res_list

    def name(self):
        return f"{self.suite.name()} {self.test}"

    def teardown(self):
        return


# multi benchmarks
class Blocked_transform(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "blocked_transform", "BlockedTransform_multi")

    def bin_args(self) -> list[str]:
        return [f"--size=2049", f"--local=1024"]


class DagTaskI(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(
            bench,
            "dag_task_throughput_independent",
            "IndependentDAGTaskThroughput_multi",
        )

    def bin_args(self) -> list[str]:
        return [
            f"--size=32768",
        ]


class DagTaskS(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(
            bench, "dag_task_throughput_sequential", "DAGTaskThroughput_multi"
        )

    def bin_args(self) -> list[str]:
        return [
            f"--size=327680",
        ]


class HostDevBandwidth(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "host_device_bandwidth", "HostDeviceBandwidth_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=512",
        ]


class LocalMem(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "local_mem", f"LocalMem_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=10240000",
        ]


class Pattern_L2(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "pattern_L2", "L2_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=1024000000",
        ]


class Reduction(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "reduction", "Pattern_Reduction_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=10240000",
        ]


class ScalarProd(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "scalar_prod", "ScalarProduct_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=102400000",
        ]


class SegmentReduction(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(
            bench, "segmentedreduction", "Pattern_SegmentedReduction_multi"
        )

    def bin_args(self) -> list[str]:
        return [
            f"--size=102400000",
        ]


class UsmAccLatency(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "usm_accessors_latency", "USM_Latency_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=4096",
        ]


class UsmAllocLatency(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(
            bench, "usm_allocation_latency", "USM_Allocation_latency_multi"
        )

    def bin_args(self) -> list[str]:
        return [
            f"--size=1024000000",
        ]


class UsmInstrMix(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "usm_instr_mix", "USM_Instr_Mix_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=8192",
        ]


class UsmPinnedOverhead(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "usm_pinned_overhead", "USM_Pinned_Overhead_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=10240000",
        ]


class VecAdd(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "vec_add", "VectorAddition_multi")

    def bin_args(self) -> list[str]:
        return [
            f"--size=102400000",
        ]


# single benchmarks
class Arith(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "arith", "Arith_int32_512")

    def bin_args(self) -> list[str]:
        return [
            f"--size=16384",
        ]


class TwoDConvolution(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "2DConvolution", "2DConvolution")


class Two_mm(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "2mm", "2mm")

    def bin_args(self) -> list[str]:
        return [
            f"--size=512",
        ]


class Three_mm(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "3mm", "3mm")

    def bin_args(self) -> list[str]:
        return [
            f"--size=512",
        ]


class Atax(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "atax", "Atax")

    def bin_args(self) -> list[str]:
        return [
            f"--size=8192",
        ]


class Atomic_reduction(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "atomic_reduction", "ReductionAtomic_fp64")


class Bicg(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "bicg", "Bicg")

    def bin_args(self) -> list[str]:
        return [
            f"--size=204800",
        ]


class Correlation(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "correlation", "Correlation")

    def bin_args(self) -> list[str]:
        return [
            f"--size=2048",
        ]


class Covariance(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "covariance", "Covariance")

    def bin_args(self) -> list[str]:
        return [
            f"--size=2048",
        ]


class Gemm(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "gemm", "Gemm")

    def bin_args(self) -> list[str]:
        return [
            f"--size=1536",
        ]


class Gesumv(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "gesummv", "Gesummv")

    def bin_args(self) -> list[str]:
        return [
            f"--size=8192",
        ]


class Gramschmidt(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "gramschmidt", "Gramschmidt")

    def bin_args(self) -> list[str]:
        return [
            f"--size=512",
        ]


class KMeans(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "kmeans", "Kmeans")

    def bin_args(self) -> list[str]:
        return [
            f"--size=700000000",
        ]


class LinRegCoeff(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "lin_reg_coeff", "LinearRegressionCoeff")

    def bin_args(self) -> list[str]:
        return [
            f"--size=1638400000",
        ]


class LinRegError(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "lin_reg_error", "LinearRegression")

    def bin_args(self) -> list[str]:
        return [
            f"--size=4096",
        ]


class MatmulChain(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "matmulchain", "MatmulChain")

    def bin_args(self) -> list[str]:
        return [
            f"--size=2048",
        ]


class MolDyn(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "mol_dyn", "MolecularDynamics")

    def bin_args(self) -> list[str]:
        return [
            f"--size=8196",
        ]


class Mvt(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "mvt", "Mvt")

    def bin_args(self) -> list[str]:
        return [
            f"--size=32767",
        ]


class NBody(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "nbody", "NBody_")

    def bin_args(self) -> list[str]:
        return [
            f"--size=81920",
        ]


class Sf(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "sf", "sf_16")

    def bin_args(self) -> list[str]:
        return [
            f"--size=5000000000",
        ]


class Syr2k(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "syr2k", "Syr2k")

    def bin_args(self) -> list[str]:
        return [
            f"--size=2048",
        ]


class Syrk(SyclBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "syrk", "Syrk")

    def bin_args(self) -> list[str]:
        return [
            f"--size=1024",
        ]
