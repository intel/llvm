/// Tests for -f[no-]sycl-rdc with --offload-new-driver.

// At the link step, -fno-sycl-rdc is silently ignored (same as upstream):
// --no-sycl-rdc is NOT forwarded to clang-linker-wrapper.
// At the compile step (-c), --no-sycl-rdc and --sycl-device-link ARE passed
// for per-TU device finalization.

// Default (no flag): RDC is ON by default for SYCL, so --no-sycl-rdc should NOT appear.
// RUN: %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl --no-offloadlib -fno-sycl-instrument-device-code %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEFAULT %s
// CHK-DEFAULT-NOT: --no-sycl-rdc

// -fno-sycl-rdc at link step: silently ignored, --no-sycl-rdc should NOT appear.
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-rdc --no-offloadlib -fno-sycl-instrument-device-code %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-RDC %s
// CHK-NO-RDC-NOT: clang-linker-wrapper{{.*}} "--no-sycl-rdc"

// AOT Intel GPU target, default RDC: --no-sycl-rdc should NOT appear.
// RUN: %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=intel_gpu_pvc --no-offloadlib -fno-sycl-instrument-device-code %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-AOT-RDC %s
// CHK-AOT-RDC-NOT: --no-sycl-rdc

// AOT Intel GPU target + -fno-sycl-rdc at link step: silently ignored, --no-sycl-rdc should NOT appear.
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=intel_gpu_pvc -fno-sycl-rdc --no-offloadlib -fno-sycl-instrument-device-code %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-AOT-NO-RDC %s
// CHK-AOT-NO-RDC-NOT: clang-linker-wrapper{{.*}} "--no-sycl-rdc"

// -fno-sycl-rdc -flto -c: per-TU device link still happens, --sycl-device-link present.
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-rdc -flto --no-offloadlib -fno-sycl-instrument-device-code -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-RDC-LTO-C %s
// CHK-NO-RDC-LTO-C: clang-linker-wrapper{{.*}} "--no-sycl-rdc"{{.*}} "--sycl-device-link"

// -fno-sycl-rdc -flto (link step): silently ignored, --no-sycl-rdc and --sycl-device-link should NOT appear.
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-rdc -flto --no-offloadlib -fno-sycl-instrument-device-code %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-RDC-LTO %s
// CHK-NO-RDC-LTO-NOT: clang-linker-wrapper{{.*}} "--no-sycl-rdc"
// CHK-NO-RDC-LTO-NOT: --sycl-device-link

// Verify pipeline with --offload-new-driver -fno-sycl-rdc.
// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-rdc %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s --check-prefix=CHECK-PIPELINE

// CHECK-PIPELINE: 0: input, "{{.*}}1.cpp", c++, (host-sycl)
// CHECK-PIPELINE: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHECK-PIPELINE: 2: compiler, {1}, ir, (host-sycl)
// CHECK-PIPELINE: 3: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK-PIPELINE: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHECK-PIPELINE: 5: compiler, {4}, ir, (device-sycl)
// CHECK-PIPELINE: 6: backend, {5}, ir, (device-sycl)
// CHECK-PIPELINE: 7: offload, "device-sycl (spir64-unknown-unknown)" {6}, ir
// CHECK-PIPELINE: 8: llvm-offload-binary, {7}, image, (device-sycl)
// CHECK-PIPELINE: 9: clang-linker-wrapper, {8}, image, (device-sycl)
// CHECK-PIPELINE: 10: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (x86_64-unknown-linux-gnu)" {9}, ir
// CHECK-PIPELINE: 11: backend, {10}, assembler, (host-sycl)
// CHECK-PIPELINE: 12: assembler, {11}, object, (host-sycl)
// CHECK-PIPELINE: 13: input, "{{.*}}2.cpp", c++, (host-sycl)
// CHECK-PIPELINE: 14: preprocessor, {13}, c++-cpp-output, (host-sycl)
// CHECK-PIPELINE: 15: compiler, {14}, ir, (host-sycl)
// CHECK-PIPELINE: 16: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK-PIPELINE: 17: preprocessor, {16}, c++-cpp-output, (device-sycl)
// CHECK-PIPELINE: 18: compiler, {17}, ir, (device-sycl)
// CHECK-PIPELINE: 19: backend, {18}, ir, (device-sycl)
// CHECK-PIPELINE: 20: offload, "device-sycl (spir64-unknown-unknown)" {19}, ir
// CHECK-PIPELINE: 21: llvm-offload-binary, {20}, image, (device-sycl)
// CHECK-PIPELINE: 22: clang-linker-wrapper, {21}, image, (device-sycl)
// CHECK-PIPELINE: 23: offload, "host-sycl (x86_64-unknown-linux-gnu)" {15}, "device-sycl (x86_64-unknown-linux-gnu)" {22}, ir
// CHECK-PIPELINE: 24: backend, {23}, assembler, (host-sycl)
// CHECK-PIPELINE: 25: assembler, {24}, object, (host-sycl)
// CHECK-PIPELINE: 26: clang-linker-wrapper, {12, 25}, image, (host-sycl)
