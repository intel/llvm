/// Tests for -fno-sycl-rdc at the compile step (-c) with --offload-new-driver.
/// Verifies that the driver inserts a per-TU clang-linker-wrapper finalize
/// action and routes the result to -foffload-include-binary on the host
/// cc1, instead of deferring device processing to link time.

// -fno-sycl-rdc -c: per-TU linker-wrapper with --sycl-device-link, host cc1 gets -foffload-include-binary.
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu \
// RUN:   --sysroot=%S/Inputs/SYCL --no-offloadlib -fno-sycl-instrument-device-code \
// RUN:   -fsycl -fno-sycl-rdc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-COMPILE --implicit-check-not=-fembed-offload-object %s
// CHK-COMPILE: clang-linker-wrapper{{.*}} "--no-sycl-rdc"{{.*}} "--sycl-device-link"
// CHK-COMPILE: "-fsycl-is-host"{{.*}} "-foffload-include-binary"

// Default RDC -c: -fembed-offload-object appears, no -foffload-include-binary or --no-sycl-rdc.
// RUN: %clang -### --offload-new-driver --target=x86_64-unknown-linux-gnu \
// RUN:   --sysroot=%S/Inputs/SYCL --no-offloadlib -fno-sycl-instrument-device-code \
// RUN:   -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RDC %s
// CHK-RDC: -fembed-offload-object
// CHK-RDC-NOT: -foffload-include-binary
// CHK-RDC-NOT: --no-sycl-rdc

// -fno-sycl-rdc at link step: --no-sycl-rdc forwarded, --sycl-device-link must NOT appear.
// RUN: touch %t.o
// RUN: %clang -### --offload-new-driver -Werror --target=x86_64-unknown-linux-gnu \
// RUN:   --sysroot=%S/Inputs/SYCL -fsycl -fno-sycl-rdc %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// CHK-LINK: clang-linker-wrapper{{.*}} "--no-sycl-rdc"
// CHK-LINK-NOT: clang-linker-wrapper{{.*}} "--sycl-device-link"
