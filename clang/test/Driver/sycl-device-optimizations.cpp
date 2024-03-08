/// Check that optimizations for sycl device are enabled by default:
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fno-sycl-early-optimizations"
// CHECK-DEFAULT-NOT: "-disable-llvm-passes"
// CHECK-DEFAULT: "-fsycl-is-device"
// CHECK-DEFAULT-SAME: "-O2"

/// Check "-fno-sycl-early-optimizations" is passed to the front-end:
// RUN:   %clang -### -fsycl -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fsycl -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang -### -fsycl -fsycl-device-only -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fsycl -fsycl-device-only -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang -### -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fintelfpga %s 2>&1 \
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
// CHECK-NO-DAE: sycl-post-link{{.*}} "-emit-param-info"

// Check "-fgpu-inline-threshold" is passed to the front-end:
// RUN:   %clang -### -fsycl -fgpu-inline-threshold=100000 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THRESH %s
// CHECK-THRESH: "-mllvm" "-inline-threshold=100000"

// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-THRESH %s
// CHECK-NO-THRESH-NOT: "-mllvm" "-inline-threshold

/// Check that optimizations for sycl device are disabled with -g passed:
// RUN:   %clang -### -fsycl -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEBUG %s
// RUN:   %clang_cl -### -fsycl -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEBUG %s
// CHECK-DEBUG: clang{{.*}} "-O0"
// CHECK-DEBUG: sycl-post-link{{.*}} "-O0"
// CHECK-DEBUG-NOT: "-O2"

/// Check that optimizations for sycl device are enabled with -g and O2 passed:
// RUN:   %clang -### -fsycl -O2 -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEBUG-O2 %s
// RUN:   %clang_cl -### -fsycl -O2 -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEBUG-O2 %s
// CHECK-DEBUG-O2: clang{{.*}} "-O2"
// CHECK-DEBUG-O2: sycl-post-link{{.*}} "-O2"
// CHECK-DEBUG-O2-NOT: "-O0"

/// Check that -O0 is passed for FPGA as it uses -g by default
// RUN:   %clang -### -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA %s
// RUN:   %clang_cl -### -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA %s
// CHECK-FPGA: clang{{.*}} "-O0"
// CHECK-FPGA: sycl-post-link{{.*}} "-O0"
// CHECK-FPGA-NOT: "-O2"

/// Check that -O" preserves for FPGA when it's explicitly passed
// RUN:   %clang -### -O2 -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA-O2 %s
// RUN:   %clang_cl -### -O2 -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA-O2 %s
// CHECK-FPGA-O2: clang{{.*}} "-O2"
// CHECK-FPGA-O2: sycl-post-link{{.*}} "-O2"
// CHECK-FPGA-O2-NOT: "-O0"
