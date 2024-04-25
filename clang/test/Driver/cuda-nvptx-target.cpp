// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-windows-msvc %s 2> %t.win.out
// RUN: FileCheck %s --check-prefixes=CHECK-WINDOWS --input-file %t.win.out
// CHECK-WINDOWS: remangled-l32-signed_char.libspirv-nvptx64-nvidia-cuda.bc
//
// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-linux-gnu %s 2> %t.lnx.out
// RUN: FileCheck %s --check-prefixes=CHECK-LINUX --input-file %t.lnx.out
// CHECK-LINUX: remangled-l64-signed_char.libspirv-nvptx64-nvidia-cuda.bc

/// Check whether an invalid SYCL target for NVidia GPU is specified:
// RUN:   not %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET-CUDA %s
// RUN:   not %clang_cl -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET-CUDA %s
// CHK-INVALID-TARGET-CUDA: error: SYCL target is invalid: 'nvptx64-nvidia-cuda-sycldevice'
