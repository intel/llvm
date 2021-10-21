/// Test that `-fsycl-libspirv-path=` produces a diagnostic when the library is not found.
// REQUIRES: clang-driver
// UNSUPPORTED: system-windows

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc %s 2>&1 \
// RUN: | FileCheck --check-prefix=ERR-CUDA %s
// ERR-CUDA: cannot find suitable 'libspirv-nvptx64--nvidiacl.bc' variant

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc %s 2>&1 \
// RUN: | FileCheck --check-prefix=ERR-HIP %s
// ERR-HIP: cannot find 'remangled-l64-signed_char.libspirv-amdgcn--amdhsa.bc'

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc -fno-sycl-libspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=OK-CUDA %s
// OK-CUDA-NOT: cannot find suitable 'libspirv-nvptx64--nvidiacl.bc' variant

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc -fno-sycl-libspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=OK-HIP %s
// OK-HIP-NOT: cannot find 'remangled-l64-signed_char.libspirv-amdgcn--amdhsa.bc'
