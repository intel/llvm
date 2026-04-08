// REQUIRES: nvptx-registered-target

// Test default behavior (no flag specified - should not add NVVM flag)
// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// Test new normalized option with different values
// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-id-queries-range=int %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-INT %s

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-id-queries-range=uint %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-UINT %s

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-id-queries-range=size_t %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// Test legacy compatibility flags
// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-id-queries-fit-in-int %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-INT %s

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fno-sycl-id-queries-fit-in-int %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// Test precedence: last one wins
// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-id-queries-range=uint -fsycl-id-queries-fit-in-int %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-INT %s

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-id-queries-fit-in-int -fsycl-id-queries-range=size_t %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// CHECK-INT: "-mllvm" "-nvvm-reflect-add=__CUDA_ID_QUERIES_FIT_IN_INT=1"
// CHECK-INT-NOT: "-nvvm-reflect-add=__CUDA_ID_QUERIES_FIT_IN_UINT=1"
// CHECK-UINT: "-mllvm" "-nvvm-reflect-add=__CUDA_ID_QUERIES_FIT_IN_UINT=1"
// CHECK-UINT-NOT: "-nvvm-reflect-add=__CUDA_ID_QUERIES_FIT_IN_INT=1"
// CHECK-DEFAULT-NOT: "-nvvm-reflect-add=__CUDA_ID_QUERIES_FIT_IN_INT=1"
// CHECK-DEFAULT-NOT: "-nvvm-reflect-add=__CUDA_ID_QUERIES_FIT_IN_UINT=1"
