/// Tests for -f[no-]sycl-rdc with --offload-new-driver.

// Verifies that --no-sycl-rdc is propagated to clang-linker-wrapper when
// -fno-sycl-rdc is passed. RDC is ON by default; --no-sycl-rdc signals
// RDC is OFF.
// UNSUPPORTED: system-windows

// RUN: touch %t.cpp

// Default (no flag): RDC is ON by default for SYCL, so --no-sycl-rdc should NOT appear.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEFAULT %s
// CHK-DEFAULT-NOT: --no-sycl-rdc

// -fno-sycl-rdc: --no-sycl-rdc should appear.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver -fno-sycl-rdc %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-RDC %s
// CHK-NO-RDC: clang-linker-wrapper{{.*}} "--no-sycl-rdc"

// AOT Intel GPU target, default RDC: --no-sycl-rdc should NOT appear.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver -fsycl-targets=intel_gpu_pvc %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-AOT-RDC %s
// CHK-AOT-RDC-NOT: --no-sycl-rdc

// AOT Intel GPU target + -fno-sycl-rdc: --no-sycl-rdc should appear.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver -fsycl-targets=intel_gpu_pvc \
// RUN:   -fno-sycl-rdc %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-AOT-NO-RDC %s
// CHK-AOT-NO-RDC: clang-linker-wrapper{{.*}} "--no-sycl-rdc"


// Verify pipeline with --offload-new-driver -fno-sycl-rdc.
// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl --offload-new-driver -fno-sycl-rdc %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s --check-prefix=CHECK-PIPELINE

// CHECK-PIPELINE: 0: input, "{{.*}}1.cpp", c++, (host-sycl)
// CHECK-PIPELINE: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHECK-PIPELINE: 2: compiler, {1}, ir, (host-sycl)
// CHECK-PIPELINE: 3: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK-PIPELINE: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHECK-PIPELINE: 5: compiler, {4}, ir, (device-sycl)
// CHECK-PIPELINE: 6: backend, {5}, ir, (device-sycl)
// CHECK-PIPELINE: 7: offload, "device-sycl (spir64-unknown-unknown)" {6}, ir
// CHECK-PIPELINE: 8: llvm-offload-binary, {7}, image, (device-sycl)
// CHECK-PIPELINE: 9: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (x86_64-unknown-linux-gnu)" {8}, ir
// CHECK-PIPELINE: 10: backend, {9}, assembler, (host-sycl)
// CHECK-PIPELINE: 11: assembler, {10}, object, (host-sycl)
// CHECK-PIPELINE: 12: input, "{{.*}}2.cpp", c++, (host-sycl)
// CHECK-PIPELINE: 13: preprocessor, {12}, c++-cpp-output, (host-sycl)
// CHECK-PIPELINE: 14: compiler, {13}, ir, (host-sycl)
// CHECK-PIPELINE: 15: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK-PIPELINE: 16: preprocessor, {15}, c++-cpp-output, (device-sycl)
// CHECK-PIPELINE: 17: compiler, {16}, ir, (device-sycl)
// CHECK-PIPELINE: 18: backend, {17}, ir, (device-sycl)
// CHECK-PIPELINE: 19: offload, "device-sycl (spir64-unknown-unknown)" {18}, ir
// CHECK-PIPELINE: 20: llvm-offload-binary, {19}, image, (device-sycl)
// CHECK-PIPELINE: 21: offload, "host-sycl (x86_64-unknown-linux-gnu)" {14}, "device-sycl (x86_64-unknown-linux-gnu)" {20}, ir
// CHECK-PIPELINE: 22: backend, {21}, assembler, (host-sycl)
// CHECK-PIPELINE: 23: assembler, {22}, object, (host-sycl)
// CHECK-PIPELINE: 24: clang-linker-wrapper, {11, 23}, image, (host-sycl)
