/// Test that appropriate warnings are output when -fno-sycl-libspirv is used.

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -fno-sycl-libspirv %s -### 2>&1 | FileCheck %s
// CHECK: warning: no-sycl-libspirv option is not meant for regular application development and severely limits the correct behavior of DPC++ when using the 'nvptx64-nvidia-cuda' triple
// CHECK: warning: no-sycl-libspirv option is not meant for regular application development and severely limits the correct behavior of DPC++ when using the 'amdgcn-amd-amdhsa' triple
// RUN: %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown -fno-sycl-libspirv %s -### 2>&1 | FileCheck --check-prefix=CHECK-SPIR64 %s
// CHECK-SPIR64: warning: no-sycl-libspirv option has no effect when compiled with the 'spir64-unknown-unknown' triple