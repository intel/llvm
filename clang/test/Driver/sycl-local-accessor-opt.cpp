/// Check the correct handling of sycl-enable-local-accessor option.

// REQUIRES: clang-driver

// RUN:   %clang -fsycl -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OPT-CHECK %s

// RUN:   %clang -fsycl -S -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OPT-CHECK %s
// OPT-CHECK-NOT: "-sycl-enable-local-accessor"

// RUN:   %clang -fsycl -fsycl-targets=nvptx64-nvidia-cuda -### %s 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-sycl-enable-local-accessor"
