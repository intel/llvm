# This script provides the complete set of all possible features that can be
# used in XFAIL, UNSUPPORTED and REQUIRES.

# To use:
# from sycl_lit_features.py import get_all_sycl_lit_features
# all_features = get_all_sycl_lit_features()

# Note:
# The set below (partial_set_of_features) is maintained manually. If the new
# feature is NOT an aspect or an architecture - it should be added to this set.
# And vice versa - if the feature was deleted, it also should be deleted from
# this set, otherwise the feature is treated as valid. Aspects and device
# architectures are added automatically and require no additional changes.

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


def get_all_sycl_lit_features():
    aspects = parse_defines(
        "../include/sycl/info/aspects.def", "__SYCL_ASPECT", "aspect"
    )
    aspects_deprecated = parse_defines(
        "../include/sycl/info/aspects_deprecated.def", "__SYCL_ASPECT", "aspect"
    )
    architectures = parse_defines(
        "../include/sycl/ext/oneapi/experimental/device_architecture.def",
        "__SYCL_ARCHITECTURE",
        "arch",
    )

    all_features = (
        partial_set_of_features | aspects | aspects_deprecated | architectures
    )
    return all_features
