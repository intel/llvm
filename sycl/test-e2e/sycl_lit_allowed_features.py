# This script provides the complete set of all possible features that can be
# used in XFAIL, UNSUPPORTED and REQUIRES.

# To use:
# from sycl_lit_allowed_features import get_sycl_lit_allowed_features
# allowed_features = get_sycl_lit_allowed_features()

# Note:
# The set below (partial_set_of_features) is maintained manually. If the new
# feature is NOT an aspect or an architecture - it should be added to this set.
# And vice versa - if the feature was deleted, it also should be deleted from
# this set, otherwise the feature is still treated as valid.
#
# Aspects and device architectures are added automatically and require no
# additional changes.

import os

partial_set_of_features = {
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
    "opencl",
    "level_zero",
    "level-zero",
    "native_cpu",
    # target:
    "target-nvidia",
    "target-native_cpu",
    "target-amd",
    "target-spir",
    "spirv-backend",
    # tools:
    "sycl-ls",
    "cm-compiler",
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
    # any-target-is-:
    "any-target-is-spir",
    "any-target-is-native_cpu",
    "any-target-is-nvidia",
    "any-target-is-amd",
    "any-target-is-native_cpu",
    # sg-sizes (should we allow any sg-X?):
    "sg-8",
    "sg-16",
    "sg-32",
    # e2e-modes:
    "run-mode",
    "build-and-run-mode",
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
    "enable-perf-tests",
}


# To parse .def files to get aspects and architectures
def parse_defines(path, macro, prefix):
    features = set()
    with open(path, "r") as file:
        for line in file:
            if line.startswith(macro):
                feature = line.split("(")[1].split(",")[0].strip()
                features.add(f"{prefix}-{feature}")
        return features


def get_sycl_lit_allowed_features():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    aspects = parse_defines(
        current_dir + "/../include/sycl/info/aspects.def", "__SYCL_ASPECT", "aspect"
    )
    aspects_deprecated = parse_defines(
        current_dir + "/../include/sycl/info/aspects_deprecated.def",
        "__SYCL_ASPECT",
        "aspect",
    )
    architectures = parse_defines(
        current_dir
        + "/../include/sycl/ext/oneapi/experimental/device_architecture.def",
        "__SYCL_ARCHITECTURE",
        "arch",
    )

    # Combine all sets
    all_features = (
        partial_set_of_features | aspects | aspects_deprecated | architectures
    )
    return all_features
