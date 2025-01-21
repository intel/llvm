// REQUIRES: amdgpu-registered-target

/// Verify that compiler passes are correctly determined
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx600 -nogpulib -c -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=CHECK-PHASES
// CHECK-PHASES: offload, "device-sycl (amdgcn-amd-amdhsa:gfx600)"

/// Verify that preprocessor works (#8112)
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx600 -nogpulib -E %s 2>&1 | FileCheck %s --check-prefix=CHECK-PREPROCESSOR
// CHECK-PREPROCESSOR: // __CLANG_OFFLOAD_BUNDLE____START__ sycl-amdgcn-amd-amdhsa-gfx600

