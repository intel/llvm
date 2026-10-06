/// Tests for -f[no-]sycl-rdc with --offload-new-driver.

// Verifies that --no-sycl-rdc is propagated to clang-linker-wrapper when
// -fno-sycl-rdc is passed. RDC is ON by default; --no-sycl-rdc signals
// RDC is OFF.

// RUN: touch %t.cpp

// Default (no flag): RDC is ON by default for SYCL, so --no-sycl-rdc should NOT appear.
// RUN: %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl --no-offloadlib -fno-sycl-instrument-device-code %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEFAULT %s
// CHK-DEFAULT-NOT: --no-sycl-rdc

// -fno-sycl-rdc: --no-sycl-rdc should appear.
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-rdc --no-offloadlib -fno-sycl-instrument-device-code %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-RDC %s
// CHK-NO-RDC: clang-linker-wrapper{{.*}} "--no-sycl-rdc"

// AOT Intel GPU target, default RDC: --no-sycl-rdc should NOT appear.
// RUN: %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=intel_gpu_pvc --no-offloadlib -fno-sycl-instrument-device-code %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-AOT-RDC %s
// CHK-AOT-RDC-NOT: --no-sycl-rdc

// AOT Intel GPU target + -fno-sycl-rdc: --no-sycl-rdc should appear.
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=intel_gpu_pvc -fno-sycl-rdc --no-offloadlib -fno-sycl-instrument-device-code %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-AOT-NO-RDC %s
// CHK-AOT-NO-RDC: clang-linker-wrapper{{.*}} "--no-sycl-rdc"

// Test compilation step: with -c, -fno-sycl-rdc finalizes the device code of
// the translation unit into a fat binary at compile time.
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen -fno-sycl-rdc --no-offloadlib -fno-sycl-instrument-device-code %t.cpp -c -o %t.o 2>&1 \
// RUN:    | FileCheck -check-prefix=CHK-COMPILE-STEP %s

// CHK-COMPILE-STEP-NOT: argument unused during compilation
// CHK-COMPILE-STEP: clang-linker-wrapper{{.*}} "--no-sycl-rdc"{{.*}} "--emit-fatbin-only"
// CHK-COMPILE-STEP: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-foffload-include-binary"

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
// CHECK-PIPELINE: 9: clang-linker-wrapper, {8}, sycl-fatbin, (device-sycl)
// CHECK-PIPELINE: 10: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown)" {9}, ir
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
// CHECK-PIPELINE: 22: clang-linker-wrapper, {21}, sycl-fatbin, (device-sycl)
// CHECK-PIPELINE: 23: offload, "host-sycl (x86_64-unknown-linux-gnu)" {15}, "device-sycl (spir64-unknown-unknown)" {22}, ir
// CHECK-PIPELINE: 24: backend, {23}, assembler, (host-sycl)
// CHECK-PIPELINE: 25: assembler, {24}, object, (host-sycl)
// CHECK-PIPELINE: 26: clang-linker-wrapper, {12, 25}, image, (host-sycl)

// -fno-sycl-rdc is rejected for Native CPU and -fsycl-embed-ir, which need
// host objects that the compile-step embedding cannot carry.
// RUN: not %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -fno-sycl-rdc -fno-sycl-libspirv --no-offloadlib -c %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NATIVE-CPU %s
// CHK-NATIVE-CPU: error: '-fno-sycl-rdc' is not supported with '-fsycl-targets=native_cpu' when using the new offloading model
// RUN: not %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a -nogpulib -fno-sycl-libspirv -fsycl-embed-ir -fno-sycl-rdc --no-offloadlib -c %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMBED-IR %s
// CHK-EMBED-IR: error: '-fno-sycl-rdc' is not supported with '-fsycl-embed-ir' when using the new offloading model

// RDC and the old offloading model are not affected.
// RUN: %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -fno-sycl-libspirv --no-offloadlib -c %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NATIVE-CPU-OK %s
// RUN: %clang -### --no-offload-new-driver --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -fno-sycl-rdc -fno-sycl-libspirv --no-offloadlib -c %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NATIVE-CPU-OK %s
// CHK-NATIVE-CPU-OK-NOT: is not supported with
