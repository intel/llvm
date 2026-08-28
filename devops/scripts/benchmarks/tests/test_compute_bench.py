import os
import sys
import unittest

sys.path.append(f"{os.path.dirname(__file__)}/../")
from benches.compute.compute import ComputeBench
from options import options


def _sin_kernel_graph_sycl_cases(ur_adapter: str, device_architecture: str) -> list[str]:
    old_adapter = options.ur_adapter
    old_arch = getattr(options, "device_architecture", None)
    try:
        options.ur_adapter = ur_adapter
        options.device_architecture = device_architecture
        return [
            b.name()
            for b in ComputeBench().benchmarks()
            if "graph_api_benchmark_sycl SinKernelGraph graphs:1" in b.name()
        ]
    finally:
        options.ur_adapter = old_adapter
        options.device_architecture = old_arch


class TestComputeBench(unittest.TestCase):
    def test_sycl_sin_kernel_graph_mode_skipped_on_pvc(self):
        cases = _sin_kernel_graph_sycl_cases("level_zero_v2", "pvc")
        self.assertFalse(cases)

    def test_sycl_sin_kernel_graph_mode_kept_on_non_pvc(self):
        cases = _sin_kernel_graph_sycl_cases("level_zero_v2", "bmg")
        self.assertEqual(
            sorted(cases),
            sorted(
                [
                    "graph_api_benchmark_sycl SinKernelGraph graphs:1, numKernels:5",
                    "graph_api_benchmark_sycl SinKernelGraph graphs:1, numKernels:100",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
