/// Test that `-fsycl-libspirv-path=` adds `-mlink-builtin-bitcode` when the library is found.
// UNSUPPORTED: system-windows

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck %s
// CHECK: {{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc" {{.*}}
