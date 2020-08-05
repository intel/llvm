/// Check that optimizations for sycl device are enabled by default:
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fno-sycl-std-optimizations"
// CHECK-DEFAULT-NOT: "-disable-llvm-passes"

/// Check "-fno-sycl-std-optimizations" is passed to the front-end:
// RUN:   %clang -### -fsycl -fno-sycl-std-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-STD-OPTS %s
// RUN:   %clang -### -fsycl -fsycl-device-only -fno-sycl-std-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-STD-OPTS %s
// CHECK-NO-SYCL-STD-OPTS: "-fno-sycl-std-optimizations"