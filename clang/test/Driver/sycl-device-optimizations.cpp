/// Check that optimizations for sycl device are enabled by default:
// RUN:   %clang -### -fsycl --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang -### -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fno-sycl-early-optimizations"
// CHECK-DEFAULT-NOT: "-disable-llvm-passes"
// CHECK-DEFAULT: "-fsycl-is-device"
// CHECK-DEFAULT-SAME: "-O2"

/// Check "-fno-sycl-early-optimizations" is passed to the front-end:
// RUN:   %clang -### -fsycl --offload-new-driver -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-device-only -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-device-only -fno-sycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang -### -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// RUN:   %clang_cl -### -fintelfpga %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-EARLY-OPTS %s
// CHECK-NO-SYCL-EARLY-OPTS: "-fno-sycl-early-optimizations"

/// Check that Dead Parameter Elimination Optimization is enabled
// RUN:   %clang -### -fsycl --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DAE %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DAE %s
// CHECK-DAE: clang{{.*}} "-fenable-sycl-dae"

/// Check that Dead Parameter Elimination Optimization is disabled
// RUN:   %clang -### -fsycl --offload-new-driver -fno-sycl-dead-args-optimization %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-DAE %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fno-sycl-dead-args-optimization %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-DAE %s
// CHECK-NO-DAE-NOT: clang{{.*}} "-fenable-sycl-dae"

// Check "-fgpu-inline-threshold" is passed to the front-end:
// RUN:   %clang -### -fsycl --offload-new-driver -fgpu-inline-threshold=100000 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THRESH %s
// CHECK-THRESH: "-mllvm" "-inline-threshold=100000"

// RUN:   %clang -### -fsycl --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-THRESH %s
// CHECK-NO-THRESH-NOT: "-mllvm" "-inline-threshold
