/// Test that `-fsycl-libspirv-path=` produces a diagnostic when the library is not found.
// REQUIRES: clang-driver
// UNSUPPORTED: system-windows

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc %s 2>&1 \
// RUN: | FileCheck --check-prefix=ERR %s
// ERR: cannot find 'libspirv-nvptx64--nvidiacl.bc'

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc -fno-sycl-libspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=OK %s
// OK-NOT: cannot find 'libspirv-nvptx64--nvidiacl.bc'
