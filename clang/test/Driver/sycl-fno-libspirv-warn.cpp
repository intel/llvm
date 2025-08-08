/// Test that appropriate warnings are output when -fno-sycl-libspirv is used.

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx908 -nogpulib -fno-sycl-libspirv %s -### 2>&1 | FileCheck %s
// CHECK-DAG: warning: '-fno-sycl-libspirv' should not be used with target 'nvptx64-nvidia-cuda'; libspirv is required for correct behavior [-Wunsafe-libspirv-not-linked]
// CHECK-DAG: warning: '-fno-sycl-libspirv' should not be used with target 'amdgcn-amd-amdhsa'; libspirv is required for correct behavior [-Wunsafe-libspirv-not-linked]
// RUN: %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown -fno-sycl-libspirv %s -### 2>&1 | FileCheck --check-prefix=CHECK-SPIR64 %s
// CHECK-SPIR64: argument unused during compilation: '-fno-sycl-libspirv' [-Wunused-command-line-argument]
