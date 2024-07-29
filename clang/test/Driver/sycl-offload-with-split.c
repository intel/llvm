///
/// Device code split specific test.
///

// REQUIRES: x86-registered-target

/// ###########################################################################

/// Ahead of Time compilation for gen, cpu - tool invocation
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT
// CHK-TOOLS-AOT: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-TOOLS-AOT: clang-offload-packager{{.*}}
// CHK-TOOLS-AOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu"
// CHK-TOOLS-AOT: clang-linker-wrapper{{.*}} {{.*}}--sycl-post-link-options=-split=auto

/// ###########################################################################

// Check -fsycl-device-code-split=per_kernel option passing.
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// CHK-ONE-KERNEL: clang-linker-wrapper{{.*}} {{.*}}--sycl-post-link-options=-split=kernel

// Check -fsycl-device-code-split=per_source option passing.
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-device-code-split=per_source %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PER-SOURCE
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-device-code-split=per_source %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PER-SOURCE
// CHK-PER-SOURCE: clang-linker-wrapper{{.*}} {{.*}}--sycl-post-link-options=-split=source

// Check -fsycl-device-code-split option passing.
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-device-code-split %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-device-code-split %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-device-code-split=auto %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-device-code-split=auto %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// CHK-AUTO: clang-linker-wrapper{{.*}} {{.*}}--sycl-post-link-options=-split=auto
