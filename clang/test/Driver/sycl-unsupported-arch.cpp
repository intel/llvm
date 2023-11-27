/// Verify that compiler passes are correctly determined
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_30 -c -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK-PHASES
// CHECK-PHASES: offload, "device-sycl (nvptx64-nvidia-cuda:sm_30)"

/// Verify that preprocessor works (#8112)
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_30 -E %s 2>&1 | FileCheck %s --check-prefix=CHECK-PREPROCESSOR
// CHECK-PREPROCESSOR: // __CLANG_OFFLOAD_BUNDLE____START__ sycl-nvptx64-nvidia-cuda-sm_30

