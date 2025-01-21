/// Test that appropriate warnings are output when -fno-sycl-libspirv is used.

// RUN: not %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -fno-sycl-libspirv %s -### 2>&1 | FileCheck %s
// CHECK-DAG: warning: '-fno-sycl-libspirv' should not be used with target 'nvptx64-nvidia-cuda'; libspirv is required for correct behavior [-Wno-libspirv-hip-cuda]
// CHECK-DAG: warning: '-fno-sycl-libspirv' should not be used with target 'amdgcn-amd-amdhsa'; libspirv is required for correct behavior [-Wno-libspirv-hip-cuda]
// RUN: %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown -fno-sycl-libspirv %s -### 2>&1 | FileCheck --check-prefix=CHECK-SPIR64 %s
// CHECK-SPIR64: ignoring '-fno-sycl-libspirv' option as it is not currently supported for target 'spir64-unknown-unknown' [-Woption-ignored]
