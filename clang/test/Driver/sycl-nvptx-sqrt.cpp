// REQUIRES: nvptx-registered-target

// RUN: not %clang -### \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   -fsycl-fp32-prec-sqrt \
// RUN:   %s \
// RUN: 2>&1 | FileCheck --check-prefix=CHECK-CORRECT %s

// CHECK-CORRECT: "-fcuda-prec-sqrt"

// RUN: not %clang -### \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   %s \
// RUN: 2>&1 | FileCheck --check-prefix=CHECK-APPROX %s

// CHECK-APPROX-NOT: "-fcuda-prec-sqrt"

void func(){};
