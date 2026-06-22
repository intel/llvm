///
/// Tests for -fsycl --offload-host-only with the new offloading driver.
///

// REQUIRES: x86-registered-target

/// ###########################################################################
/// Test phase output with -ccc-print-phases
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --offload-new-driver --offload-host-only -ccc-print-phases -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s

// CHK-PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 2: compiler, {1}, ir, (host-sycl)
// CHK-PHASES: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES: 4: compiler, {3}, none, (device-sycl)
// CHK-PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown)" {4}, ir
// CHK-PHASES: 6: backend, {5}, assembler, (host-sycl)
// CHK-PHASES: 7: assembler, {6}, object, (host-sycl)

/// ###########################################################################
/// Test that device compile generates integration header/footer, and the host
/// compiler consumes these.
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --offload-new-driver --offload-host-only -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INT-HEADER %s

// CHK-INT-HEADER: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-INT-HEADER-SAME: "-fsycl-is-device"
// CHK-INT-HEADER-SAME: "-fsycl-int-header=[[HEADER:.+\.h]]" "-fsycl-int-footer=[[FOOTER:.+\.h]]"
// CHK-INT-HEADER-SAME: "-fsyntax-only"
// CHK-INT-HEADER: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-INT-HEADER-SAME: "-fsycl-is-host"
// CHK-INT-HEADER-SAME: "-include-internal-header" "[[HEADER]]"
// CHK-INT-HEADER-SAME: "-dependency-filter" "[[HEADER]]"
// CHK-INT-HEADER-SAME: "-include-internal-footer" "[[FOOTER]]"
// CHK-INT-HEADER-SAME: "-dependency-filter" "[[FOOTER]]"
// CHK-INT-HEADER-SAME: "-emit-obj"
// CHK-INT-HEADER-SAME: "-o" "{{.*\.o}}"

/// ###########################################################################
/// Test that no bundled output is created (only 2 invocations: device + host)
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --offload-new-driver --offload-host-only -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-BUNDLER %s

// CHK-NO-BUNDLER-NOT: clang-offload-bundler
// CHK-NO-BUNDLER-NOT: clang-offload-packager

/// ###########################################################################
/// Test with explicit target triple
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --offload-new-driver --offload-host-only -fsycl-targets=spir64 -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET %s

// CHK-TARGET: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-TARGET-SAME: "-fsycl-is-device"
// CHK-TARGET-SAME: "-fsycl-int-header=
// CHK-TARGET-SAME: "-fsyntax-only"
// CHK-TARGET: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-TARGET-SAME: "-fsycl-is-host"
// CHK-TARGET-SAME: "-include-internal-header"

/// ###########################################################################
/// Test that linking works with host-only objects
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --offload-new-driver --offload-host-only -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s

// CHK-LINK: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-LINK-SAME: "-fsycl-is-device"
// CHK-LINK-SAME: "-fsyntax-only"
// CHK-LINK: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-LINK-SAME: "-fsycl-is-host"
// CHK-LINK-SAME: "-emit-obj"
// CHK-LINK: ld{{.*}}

/// ###########################################################################
/// Test with multiple source files
// RUN: %clang -fsycl --target=x86_64-unknown-linux-gnu --offload-new-driver --offload-host-only -### -c %s %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-MULTI %s

// CHK-MULTI: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-MULTI-SAME: "-fsycl-is-device"
// CHK-MULTI-SAME: "-fsyntax-only"
// CHK-MULTI: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-MULTI-SAME: "-fsycl-is-host"
// CHK-MULTI-SAME: "-emit-obj"
// CHK-MULTI: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// CHK-MULTI-SAME: "-fsycl-is-device"
// CHK-MULTI-SAME: "-fsyntax-only"
// CHK-MULTI: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHK-MULTI-SAME: "-fsycl-is-host"
// CHK-MULTI-SAME: "-emit-obj"

/// Dummy C++ code for testing
void foo() {}
