/// Check for diagnostic when command line link targets to not match object
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/liblin64.a \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_GEN_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -L%S/Inputs/SYCL -llin64 \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_GEN_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/objlin64.o \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_GEN_DIAG
// SPIR64_GEN_DIAG: linked binaries do not contain expected 'spir64_gen-unknown-unknown' target; found targets: 'spir64-unknown-unknown{{.*}}, spir64-unknown-unknown{{.*}}' [-Wsycl-target]

// RUN: %clangxx -fsycl -fsycl-targets=spir64 %S/Inputs/SYCL/liblin64.a \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64 -L%S/Inputs/SYCL -llin64 \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64 %S/Inputs/SYCL/objlin64.o \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/liblin64.a \
// RUN:   -Wno-sycl-target -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_DIAG
// SPIR64_DIAG-NOT: linked binaries do not contain expected
