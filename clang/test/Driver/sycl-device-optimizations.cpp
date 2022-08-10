/// Check that optimizations for sycl device are enabled by default:
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang -### -fsycl -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fno-sycl-early-optimizations"
// CHECK-DEFAULT-NOT: "-disable-llvm-passes"

/// Check "-fno-sycl-early-optimizations" is passed to the front-end:
// RUN:   %clang -### -fsycl -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fsycl -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang -### -fsycl -fsycl-device-only -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fsycl -fsycl-device-only -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang -### -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fsycl -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// CHECK-NO-SYCL-EARLY-OPTS: "-fno-sycl-early-optimizations"

/// Check that Dead Parameter Elimination Optimization is enabled
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DAE %s
// RUN:   %clang_cl -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DAE %s
// CHECK-DAE: clang{{.*}} "-fenable-sycl-dae"
// CHECK-DAE: sycl-post-link{{.*}} "-emit-param-info"

/// Check that Dead Parameter Elimination Optimization is disabled
// RUN:   %clang -### -fsycl -fno-sycl-dead-args-optimization %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-DAE %s
// RUN:   %clang_cl -### -fsycl -fno-sycl-dead-args-optimization %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-DAE %s
// CHECK-NO-DAE-NOT: clang{{.*}} "-fenable-sycl-dae"
// CHECK-NO-DAE-NOT: sycl-post-link{{.*}} "-emit-param-info"

// Check "-fgpu-inline-threshold" is passed to the front-end:
// RUN:   %clang -### -fsycl -fgpu-inline-threshold=100000 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THRESH %s
// CHECK-THRESH: "-mllvm" "-inline-threshold=100000"

// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-THRESH %s
// CHECK-NO-THRESH-NOT: "-mllvm" "-inline-threshold
