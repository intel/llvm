// Tests specific to `-fsycl-targets=nvptx64-nvidia-nvptx`
// Verify that the correct devicelib linking actions are spawned by the driver.
// Check also if the correct warnings are generated.

// UNSUPPORTED: system-windows

// Check that llvm-link uses the "-only-needed" flag.
// Not using the flag breaks kernel bundles.
// RUN: %clangxx -###  --cuda-path=%S/Inputs/CUDA/usr/local/cuda -fno-sycl-libspirv --sysroot=%S/Inputs/SYCL \
// RUN: -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 | FileCheck -check-prefix=CHK-ONLY-NEEDED %s

// CHK-ONLY-NEEDED: llvm-link"{{.*}}"-only-needed"{{.*}}"{{.*}}devicelib-nvptx64-nvidia-cuda.bc"{{.*}}
