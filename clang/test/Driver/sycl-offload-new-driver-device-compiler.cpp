/// Tests that the user-facing --device-compiler= flag is forwarded verbatim
/// to clang-linker-wrapper under the new offloading driver, and that using it
/// without --offload-new-driver is diagnosed.

/// Bare value (matches any kind, any triple).
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:     --sysroot=%S/Inputs/SYCL -fsycl-targets=spirv64-unknown-unknown \
// RUN:     --device-compiler=-DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BARE %s
// CHK-BARE: clang-linker-wrapper{{.*}} "--device-compiler=-DFOO"

/// Triple-scoped value.
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:     --sysroot=%S/Inputs/SYCL -fsycl-targets=spirv64-unknown-unknown \
// RUN:     --device-compiler=spirv64-unknown-unknown=-O1 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TRIPLE %s
// CHK-TRIPLE: clang-linker-wrapper{{.*}} "--device-compiler=spirv64-unknown-unknown=-O1"

/// Kind+triple value.
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:     --sysroot=%S/Inputs/SYCL -fsycl-targets=spirv64-unknown-unknown \
// RUN:     --device-compiler=sycl:spirv64-unknown-unknown=-mllvm=-foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-KIND %s
// CHK-KIND: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spirv64-unknown-unknown=-mllvm=-foo"

/// Multiple occurrences are each forwarded.
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:     --sysroot=%S/Inputs/SYCL -fsycl-targets=spirv64-unknown-unknown \
// RUN:     --device-compiler=spirv64-unknown-unknown=-O1 \
// RUN:     --device-compiler=spirv64-unknown-unknown=-DBAR %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-MULTI %s
// CHK-MULTI: clang-linker-wrapper{{.*}} "--device-compiler=spirv64-unknown-unknown=-O1"{{.*}} "--device-compiler=spirv64-unknown-unknown=-DBAR"

/// Without --offload-new-driver the flag is rejected.
// RUN:   not %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:     --sysroot=%S/Inputs/SYCL -fsycl-targets=spirv64-unknown-unknown \
// RUN:     --device-compiler=-DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ERR %s
// CHK-ERR: error: invalid argument '--device-compiler=-DFOO' only allowed with '--offload-new-driver'
