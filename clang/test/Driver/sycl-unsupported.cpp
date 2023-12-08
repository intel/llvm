/// Diagnose unsupported options specific to SYCL compilations

// RUN: %clangxx -fsycl -fcf-protection -fsycl-targets=spir64 -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fcf-protection
// RUN: %clang_cl -fsycl -fcf-protection -fsycl-targets=spir64 -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fcf-protection
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -fcf-protection -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_gen -DOPT=-fcf-protection
// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga -fcf-protection -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_fpga -DOPT=-fcf-protection
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -fcf-protection -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_x86_64 -DOPT=-fcf-protection

// RUN: %clangxx -fsycl -fprofile-instr-generate -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fprofile-instr-generate \
// RUN:    -DOPT_CC1=-fprofile-instrument=clang \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT
// RUN: %clangxx -fsycl -fcoverage-mapping \
// RUN:          -fprofile-instr-generate -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fcoverage-mapping
// RUN: %clangxx -fsycl -ftest-coverage -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-ftest-coverage \
// RUN:    -DOPT_CC1=-coverage-notes-file \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT
// RUN: %clangxx -fsycl -fcreate-profile -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fcreate-profile \
// RUN:    -check-prefix UNSUPPORTED_OPT_DIAG
// RUN: %clangxx -fsycl -fprofile-arcs -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fprofile-arcs \
// RUN:    -DOPT_CC1=-coverage-data-file \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT
// RUN: %clangxx -fsycl -fcs-profile-generate -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fcs-profile-generate \
// RUN:    -DOPT_CC1=-fprofile-instrument=csllvm \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT
// RUN: %clangxx -fsycl -forder-file-instrumentation -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-forder-file-instrumentation
// Check to make sure our '-fsanitize=address' exception isn't triggered by a
// different option
// RUN: %clangxx -fsycl -fprofile-instr-generate=address -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fprofile-instr-generate=address \
// RUN:    -DOPT_CC1=-fprofile-instrument=clang \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT

// CHECK: ignoring '[[OPT]]' option as it is not currently supported for target '[[ARCH]]{{.*}}' [-Woption-ignored]
// CHECK-NOT: clang{{.*}} "-fsycl-is-device"{{.*}} "[[OPT]]{{.*}}"
// CHECK: clang{{.*}} "-fsycl-is-host"{{.*}} "[[OPT]]{{.*}}"

// UNSUPPORTED_OPT_DIAG: ignoring '[[OPT]]' option as it is not currently supported for target '[[ARCH]]{{.*}}' [-Woption-ignored]
// UNSUPPORTED_OPT-NOT: clang{{.*}} "-fsycl-is-device"{{.*}} "[[OPT_CC1]]{{.*}}"
// UNSUPPORTED_OPT: clang{{.*}} "-fsycl-is-host"{{.*}} "[[OPT_CC1]]{{.*}}"
