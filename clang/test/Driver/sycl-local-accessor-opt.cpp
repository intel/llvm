/// Check the correct handling of sycl-enable-local-accessor option.

// REQUIRES: clang-driver

// RUN:   %clang -fsycl -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-OPT %s
// CHECK-NO-OPT-NOT: "-sycl-enable-local-accessor"

// RUN:   %clang -fsycl -fsycl-targets=nvptx64-nvidia-cuda -### %s 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-sycl-enable-local-accessor"
