// Test the inclusion of the CUDA runtime wrapper in the SYCL device
// compilation of .cu sources.

// RUN:   %clangxx -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN::  --cuda-path=%S/Inputs/CUDA/usr/local/cuda -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-HEADERS-SYCL-CUDA %s
// CHK-HEADERS-SYCL-CUDA: clang{{.*}}"-fsycl-is-device"{{.*}}"-include" "__clang_cuda_runtime_wrapper.h"

