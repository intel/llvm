// REQUIRES: nvptx-registered-target

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fno-sycl-id-queries-fit-in-int %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-id-queries-fit-in-int %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-INT %s

// CHECK-INT: "-mllvm" "-nvvm-reflect-add=__CUDA_ID_QUERIES_FIT_IN_INT=1"
// CHECK-DEFAULT-NOT: "-nvvm-reflect-add=__CUDA_ID_QUERIES_FIT_IN_INT=1"
