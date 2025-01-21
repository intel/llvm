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

/// Check that optimizations for sycl device are disabled with -g passed:
// RUN:   %clang -### -fsycl -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEBUG %s
// RUN:   %clang_cl -### -fsycl -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEBUG %s
// CHECK-DEBUG: clang{{.*}} "-fsycl-is-device{{.*}}" "-O0"
// CHECK-DEBUG: sycl-post-link{{.*}} "-O0"
// CHECK-DEBUG-NOT: "-O2"

/// Check that optimizations for sycl device are enabled with -g and O2 passed:
// RUN:   %clang -### -fsycl -O2 -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-G-O2 %s
// For clang_cl, -O2 maps to -O3
// RUN:   %clang_cl -### -fsycl -O2 -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-G-O3 %s
// CHECK-G-O2: clang{{.*}} "-fsycl-is-device{{.*}}" "-O2"
// CHECK-G-O2: sycl-post-link{{.*}} "-O2"
// CHECK-G-O2-NOT: "-O0"
// CHECK-G-O3: clang{{.*}} "-fsycl-is-device{{.*}}" "-O3"
// CHECK-G-O3: sycl-post-link{{.*}} "-O3"
// CHECK-G-O3-NOT: "-O0"

/// Check that -O2 is passed for FPGA
// RUN:   %clang -### -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA %s
// RUN:   %clang_cl -### -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA %s
// CHECK-FPGA: clang{{.*}} "-fsycl-is-device{{.*}}" "-O2"
// CHECK-FPGA: sycl-post-link{{.*}} "-O2"
// CHECK-FPGA-NOT: "-O0"

/// Check that -O2 preserves for FPGA when it's explicitly passed
// RUN:   %clang -### -O2 -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA-O2 %s
// For clang_cl, -O2 maps to -O3
// RUN:   %clang_cl -### -O2 -fintelfpga -fsycl-early-optimizations %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA-O3 %s
// CHECK-FPGA-O2: clang{{.*}} "-fsycl-is-device{{.*}}" "-O2"
// CHECK-FPGA-O2: sycl-post-link{{.*}} "-O2"
// CHECK-FPGA-O2-NOT: "-O0"
// CHECK-FPGA-O3: clang{{.*}} "-fsycl-is-device{{.*}}" "-O3"
// CHECK-FPGA-O3: sycl-post-link{{.*}} "-O3"
// CHECK-FPGA-O3-NOT: "-O0"

/// Check that -O0 is passed for FPGA when -g is explicitly passed
// RUN:   %clang -### -fintelfpga -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA-O0 %s
// RUN:   %clang_cl -### -fintelfpga -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FPGA-O0 %s
// CHECK-FPGA-O0: clang{{.*}} "-fsycl-is-device{{.*}}" "-O0"
// CHECK-FPGA-O0: sycl-post-link{{.*}} "-O0"
// CHECK-FPGA-O0-NOT: "-O2"



