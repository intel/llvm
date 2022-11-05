///
/// Make sure that CUDA_PATH is picked up correctly when looking for CUDA
/// installation.
///

// REQUIRES: system-windows
// RUN: env CUDA_PATH=%S\Inputs\CUDA_111\usr\local\cuda %clang -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda -### -v %s 2>&1 | \
// RUN: FileCheck %s

// CHECK: Found CUDA installation: {{.*}}Inputs\CUDA_111\usr\local\cuda, version 11.1
