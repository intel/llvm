// Check that we correctly determine that the final link command links
// devicelibs together, as far as the driver is concerned. This results in the
// -only-needed flag.
//
// Note we check the names of the various device libraries because that's the
// logic the driver uses.

// Older CUDA versions had versioned libdevice files. We don't support CUDA
// this old in SYCL, but we still test the driver's ability to pick out the
// correctly versioned libdevice. We use Inputs/CUDA_80 which has a full set of
// libdevice files.
// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:    -Xsycl-target-backend --cuda-gpu-arch=sm_30 \
// RUN:    --sysroot=%S/Inputs/SYCL --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:    | FileCheck %s --check-prefixes=CHECK,LIBDEVICE30
// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:    -Xsycl-target-backend --cuda-gpu-arch=sm_35 \
// RUN:    --sysroot=%S/Inputs/SYCL --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:    | FileCheck %s --check-prefixes=CHECK,LIBDEVICE35
// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:    -Xsycl-target-backend --cuda-gpu-arch=sm_50 \
// RUN:    --sysroot=%S/Inputs/SYCL --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:    | FileCheck %s --check-prefixes=CHECK,LIBDEVICE50

// CUDA-9+ uses the same libdevice for all GPU variants
// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:    -Xsycl-target-backend --cuda-gpu-arch=sm_35 \
// RUN:    --sysroot=%S/Inputs/SYCL --cuda-path=%S/Inputs/CUDA_90/usr/local/cuda %s 2>&1 \
// RUN:    | FileCheck %s --check-prefixes=CHECK,LIBDEVICE10

// Check also that -nocudalib is obeyed
// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib \
// RUN:    -Xsycl-target-backend --cuda-gpu-arch=sm_35 \
// RUN:    --sysroot=%S/Inputs/SYCL --cuda-path=%S/Inputs/CUDA_90/usr/local/cuda %s 2>&1 \
// RUN:    | FileCheck %s --check-prefixes=CHECK,NOLIBDEVICE

// First link command: ignored
// CHECK: llvm-link

// CHECK: llvm-link
// CHECK-SAME: -only-needed
// CHECK-SAME: devicelib-nvptx64-nvidia-cuda.bc
// CHECK-SAME: libspirv-nvptx64-nvidia-cuda.bc
// LIBDEVICE10-SAME: libdevice.10.bc
// LIBDEVICE30-SAME: libdevice.compute_30.10.bc
// LIBDEVICE35-SAME: libdevice.compute_35.10.bc
// LIBDEVICE50-SAME: libdevice.compute_50.10.bc
// NOLIBDEVICE-NOT: libdevice.{{.*}}.10.bc
