/// Test that `-fsycl-libspirv-path=` produces a diagnostic when the library is not found.
// REQUIRES: clang-driver
// UNSUPPORTED: system-windows

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc %s 2>&1 \
// RUN: | FileCheck --check-prefix=ERR-CUDA %s
// ERR-CUDA: cannot find 'remangled-l64-signed_char.libspirv-nvptx64--nvidiacl.bc';
// ERR-CUDA-SAME: provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-windows-msvc -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc %s 2>&1 \
// RUN: | FileCheck --check-prefix=ERR-CUDA-WIN %s
// ERR-CUDA-WIN: cannot find 'remangled-l32-signed_char.libspirv-nvptx64--nvidiacl.bc';
// ERR-CUDA-WIN-SAME: provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc %s 2>&1 \
// RUN: | FileCheck --check-prefix=ERR-HIP %s
// ERR-HIP: cannot find 'remangled-l64-signed_char.libspirv-amdgcn--amdhsa.bc';
// ERR-HIP-SAME: provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc -fno-sycl-libspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=OK-CUDA %s
// OK-CUDA-NOT: cannot find suitable 'remangled-l64-signed_char.libspirv-nvptx64--nvidiacl.bc'

// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/no-libspirv-exists-here.bc -fno-sycl-libspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=OK-HIP %s
// OK-HIP-NOT: cannot find 'remangled-l64-signed_char.libspirv-amdgcn--amdhsa.bc'
