/// Tests for -fno-sycl-rdc at the compile step (-c) with --offload-new-driver.
/// Verifies that the driver inserts a per-TU clang-linker-wrapper finalize
/// action and routes the result to -fsycl-include-target-binary on the host
/// cc1, instead of deferring device processing to link time.

// RUN: touch %t.cpp

// --- CHECK 1: -fno-sycl-rdc -c ---
// Per-TU finalize: clang-linker-wrapper must be invoked with --sycl-device-link
// and --no-sycl-rdc. Host cc1 must receive -fsycl-include-target-binary.
// -fembed-offload-object must NOT appear (device code is not deferred).
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu \
// RUN:   -fsycl -fno-sycl-rdc -c %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-COMPILE %s
// CHK-COMPILE: clang-linker-wrapper{{.*}} "--no-sycl-rdc"{{.*}} "--sycl-device-link"
// CHK-COMPILE: "-fsycl-include-target-binary"
// CHK-COMPILE-NOT: -fembed-offload-object

// --- CHECK 2: default RDC -c (no flag) ---
// Default RDC path: -fembed-offload-object must appear (raw bitcode deferred to
// link time). -fsycl-include-target-binary and --no-sycl-rdc must NOT appear.
// RUN: %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu \
// RUN:   -fsycl -c %t.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RDC %s
// CHK-RDC: -fembed-offload-object
// CHK-RDC-NOT: -fsycl-include-target-binary
// CHK-RDC-NOT: --no-sycl-rdc

// --- CHECK 3: -fno-sycl-rdc at link step (regression for PR #22832) ---
// At link time --no-sycl-rdc is forwarded to clang-linker-wrapper.
// --sycl-device-link must NOT appear for the final link invocation.
// RUN: touch %t.o
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu \
// RUN:   -fsycl -fno-sycl-rdc %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// CHK-LINK: clang-linker-wrapper{{.*}} "--no-sycl-rdc"
// CHK-LINK-NOT: clang-linker-wrapper{{.*}} "--sycl-device-link"
