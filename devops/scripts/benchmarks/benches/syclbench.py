# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import csv
import io
from utils.utils import run, git_clone, create_build_path
from .base import Benchmark, Suite
from .result import Result
from options import options


class SyclBench(Suite):
    def __init__(self, directory):
        if options.sycl is None:
            return

        self.directory = directory
        return

    def name(self) -> str:
        return "SYCL-Bench"

    def setup(self):
        if options.sycl is None:
            return

        build_path = create_build_path(self.directory, "sycl-bench-build")
        repo_path = git_clone(
            self.directory,
            "sycl-bench-repo",
            "https://github.com/mateuszpn/sycl-bench.git",
            "1e6ab2cfd004a72c5336c26945965017e06eab71",
        )

        configure_command = [
            "cmake",
            f"-B {build_path}",
            f"-S {repo_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={options.sycl}/bin/clang++",
            f"-DCMAKE_C_COMPILER={options.sycl}/bin/clang",
            f"-DSYCL_IMPL=dpcpp",
        ]

        if options.ur_adapter == "cuda":
            configure_command += [
                f"-DCMAKE_CXX_FLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda"
            ]

        run(configure_command, add_sycl=True)
        run(f"cmake --build {build_path} -j", add_sycl=True)

        self.built = True

    def benchmarks(self) -> list[Benchmark]:
        if options.sycl is None:
            return []

        return [
            # Blocked_transform(self), # run time < 1ms
            DagTaskI(self),
            DagTaskS(self),
            HostDevBandwidth(self),
            LocalMem(self),
            Pattern_L2(self),
            Reduction(self),
            ScalarProd(self),
            SegmentReduction(self),
            UsmAccLatency(self),
            UsmAllocLatency(self),
            UsmInstrMix(self),
            UsmPinnedOverhead(self),
            VecAdd(self),
            # *** sycl-bench single benchmarks
            # TwoDConvolution(self), # run time < 1ms
            Two_mm(self),
            Three_mm(self),
            # Arith(self), # run time < 1ms
            Atax(self),
            # Atomic_reduction(self), # run time < 1ms
            Bicg(self),
            Correlation(self),
            Covariance(self),
            Gemm(self),
            Gesumv(self),
            Gramschmidt(self),
            KMeans(self),
            LinRegCoeff(self),
            # LinRegError(self), # run time < 1ms
            MatmulChain(self),
            MolDyn(self),
            Mvt(self),
            Sf(self),
            Syr2k(self),
            Syrk(self),
        ]


class SyclBenchmark(Benchmark):
    def __init__(self, bench, name, test):
        super().__init__(bench.directory, bench)
        self.bench = bench
        self.bench_name = name
        self.test = test
        self.done = False

    def bin_args(self) -> list[str]:
        return []

    def extra_env_vars(self) -> dict:
        return {}

    def setup(self):
        self.benchmark_bin = os.path.join(
            self.directory, "sycl-bench-build", self.bench_name
        )

    def run(self, env_vars) -> list[Result]:
        if self.done:
            return
        self.outputfile = os.path.join(self.bench.directory, self.test + ".csv")
        print(f"{self.__class__.__name__}: Results in {self.outputfile}")
        command = [
            f"{self.benchmark_bin}",
            f"--warmup-run",
            f"--num-runs={options.iterations}",
            f"--output={self.outputfile}",
        ]

        command += self.bin_args()
        env_vars.update(self.extra_env_vars())

        # no output to stdout, all in outputfile
        self.run_bench(command, env_vars)

        with open(self.outputfile, "r") as f:
            reader = csv.reader(f)
            res_list = []
            for row in reader:
                if not row[0].startswith("#"):
                    res_list.append(
                        Result(
                            label=row[0],
                            value=float(row[12]) * 1000,  # convert to ms
                            passed=(row[1] == "PASS"),
                            command=command,
                            env=env_vars,
                            stdout=row,
                            unit="ms",
                        )
                    )
        self.done = True
        return res_list

    def teardown(self):
        print(f"Removing {self.outputfile}...")
        os.remove(self.outputfile)
        return

    def name(self):
        return self.test


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
