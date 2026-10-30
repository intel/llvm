///
/// Tests for -fsycl --offload-host-only with the old offloading model.
///

/// ###########################################################################
/// Test phase output with -ccc-print-phases
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --no-offload-new-driver --offload-host-only -ccc-print-phases -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s

// CHK-PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// CHK-PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-PHASES: 2: compiler, {1}, ir, (device-sycl)
// CHK-PHASES: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, ir
// CHK-PHASES: 4: input, "[[INPUT]]", c++, (host-sycl)
// CHK-PHASES: 5: preprocessor, {4}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {5}, "device-sycl (spir64-unknown-unknown)" {2}, c++-cpp-output
// CHK-PHASES: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES: 10: clang-offload-bundler, {3, 9}, object, (host-sycl)

/// ###########################################################################
/// Test that device compile generates integration header/footer, and the host
/// compiler consumes these.
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --no-offload-new-driver --offload-host-only -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INT-HEADER %s

// CHK-INT-HEADER: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-INT-HEADER-SAME: "-fsycl-is-device"
// CHK-INT-HEADER-SAME: "-fsycl-int-header=[[HEADER:.+\.h]]" "-fsycl-int-footer=[[FOOTER:.+\.h]]"
// CHK-INT-HEADER-SAME: "-emit-llvm-bc" {{.*}} "-o" "[[DEVICEBC:.+\.bc]]"
// CHK-INT-HEADER: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-INT-HEADER-SAME: "-fsycl-is-host"
// CHK-INT-HEADER-SAME: "-include-internal-header" "[[HEADER]]"
// CHK-INT-HEADER-SAME: "-dependency-filter" "[[HEADER]]"
// CHK-INT-HEADER-SAME: "-include-internal-footer" "[[FOOTER]]"
// CHK-INT-HEADER-SAME: "-dependency-filter" "[[FOOTER]]"
// CHK-INT-HEADER-SAME: "-emit-obj"
// CHK-INT-HEADER-SAME: "-o" "[[HOSTOBJ:.+\.o]]"
// CHK-INT-HEADER: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown,host-x86_64-unknown-linux-gnu"
// CHK-INT-HEADER-SAME: "-input=[[DEVICEBC]]" "-input=[[HOSTOBJ]]" "-target-symbols-only"

/// ###########################################################################
/// Test with multiple source files
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --no-offload-new-driver --offload-host-only -### -c %s %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-MULTI %s

// CHK-MULTI: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-MULTI-SAME: "-fsycl-is-device"
// CHK-MULTI-SAME: "-emit-llvm-bc"
// CHK-MULTI: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-MULTI-SAME: "-fsycl-is-host"
// CHK-MULTI-SAME: "-emit-obj"
// CHK-MULTI: clang-offload-bundler{{.*}} "-target-symbols-only"
// CHK-MULTI: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-MULTI-SAME: "-fsycl-is-device"
// CHK-MULTI-SAME: "-emit-llvm-bc"
// CHK-MULTI: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-MULTI-SAME: "-fsycl-is-host"
// CHK-MULTI-SAME: "-emit-obj"
// CHK-MULTI: clang-offload-bundler{{.*}} "-target-symbols-only"

/// ###########################################################################
/// Test that device symbols are only added when generating an object.
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --no-offload-new-driver --offload-host-only -### -S %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-SYMS %s
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --no-offload-new-driver --offload-host-only -### -c -emit-llvm %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-SYMS %s
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --no-offload-new-driver --offload-host-only -### -c -flto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-SYMS %s
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --no-offload-new-driver --offload-host-only -### -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-SYMS %s

// CHK-NO-SYMS: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-NO-SYMS-SAME: "-fsycl-is-device"
// CHK-NO-SYMS-SAME: "-fsyntax-only"
// CHK-NO-SYMS: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-NO-SYMS-SAME: "-fsycl-is-host"
// CHK-NO-SYMS-NOT: clang-offload-bundler
