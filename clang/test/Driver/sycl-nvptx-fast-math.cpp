// REQUIRES: nvptx-registered-target

// RUN: %clang -### -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// RUN: %clang -### -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -ffast-math %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-FAST %s

// RUN: %clang -### -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -funsafe-math-optimizations %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-FAST %s

// CHECK-FAST: "-mllvm" "--nvptx-prec-divf32=0" "-mllvm" "--nvptx-prec-sqrtf32=0"

// CHECK-DEFAULT-NOT: "nvptx-prec-divf32=0"
// CHECK-DEFAULT-NOT: "nvptx-prec-sqrtf32=0"
