# See check-correctness-of-requirements.cpp

import re
import sys


# To parse .def files to get aspects and architectures
def parse_defines(path, macro, prefix):
    features = set()
    with open(path, "r") as file:
        for line in file:
            if line.startswith(macro):
                feature = line.split("(")[1].split(",")[0].strip()
                features.add(f"{prefix}-{feature}")
        return features


def parse_requirements(input_data_path, sycl_include_dir_path):
    available_features = {
        # for completely disabled tests
        "true",
        # host OS:
        "windows",
        "system-windows",
        "linux",
        "system-linux",
        # target device:
        "cpu",
        "gpu",
        "accelerator",
        # target backend:
        "cuda",
        "hip",
        "hip_amd",
        "hip_nvidia",
        "opencl",
        "level_zero",
        "level-zero",
        "native_cpu",
        # tools:
        "sycl-ls",
        "cm-compiler",
        "aot_tool",
        "ocloc",
        "opencl-aot",
        "llvm-spirv",
        "llvm-link",
        # dev-kits:
        "level_zero_dev_kit",
        "cuda_dev_kit",
        # manually-set features (deprecated, no new tests should use these features)
        "gpu-intel-gen11",
        "gpu-intel-gen12",
        "gpu-intel-dg1",
        "gpu-intel-dg2",
        "gpu-intel-pvc",
        "gpu-intel-pvc-vg",
        "gpu-intel-pvc-1T",
        "gpu-intel-pvc-2T",
        "gpu-amd-gfx90a",
        # any-device-is-:
        "any-device-is-cpu",
        "any-device-is-gpu",
        "any-device-is-accelerator",
        "any-device-is-cuda",
        "any-device-is-hip",
        "any-device-is-opencl",
        "any-device-is-level_zero",
        "any-device-is-native_cpu",
        # sg-sizes (should we allow any sg-X?)
        "sg-8",
        "sg-16",
        "sg-32",
        # miscellaneous:
        "cl_options",
        "opencl_icd",
        "dump_ir",
        "xptifw",
        "has_ndebug",
        "zstd",
        "preview-breaking-changes-supported",
        "vulkan",
        "O0",
        "ze_debug",
        "igc-dev",
        # Note: aspects and architectures are gathered below
    }

    available_features.update(
        parse_defines(
            sycl_include_dir_path + "/sycl/info/aspects.def", "__SYCL_ASPECT", "aspect"
        )
    )
    available_features.update(
        parse_defines(
            sycl_include_dir_path + "/sycl/info/aspects_deprecated.def",
            "__SYCL_ASPECT",
            "aspect",
        )
    )
    available_features.update(
        parse_defines(
            sycl_include_dir_path + "/sycl/ext/oneapi/experimental/architectures.def",
            "__SYCL_ARCHITECTURE",
            "arch",
        )
    )

    exit_code = 0
    with open(input_data_path, "r") as file:
        requirements = set(file.read().split())
        for requirement in requirements:
            if not requirement in available_features:
                exit_code = 1
                print("Unsupported requirement: " + requirement)
    sys.exit(exit_code)


parse_requirements(sys.argv[1], sys.argv[2])
