/// Diagnose unsupported options specific to SYCL compilations

// RUN: %clangxx -fsycl -fcf-protection -fsycl-targets=spir64 -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fcf-protection
// RUN: %clang_cl -fsycl -fcf-protection -fsycl-targets=spir64 -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fcf-protection
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -fcf-protection -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_gen -DOPT=-fcf-protection
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -fcf-protection -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_x86_64 -DOPT=-fcf-protection

// Check to make sure -gline-tables-only is passed to -fsycl-is-host invocation only.
// RUN: %clangxx -### -fsycl -gline-tables-only %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-gline-tables-only \
// RUN:    -DOPT_CC1=-debug-info-kind=line-tables-only \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT
// RUN: %clang_cl -### -fsycl -gline-tables-only %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-gline-tables-only \
// RUN:    -DOPT_CC1=-debug-info-kind=line-tables-only \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT

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
// RUN: %clangxx -fsycl --coverage -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=--coverage \
// RUN:    -DOPT_CC1=-coverage-notes-file \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT
// RUN: %clang_cl -fsycl --coverage -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=--coverage \
// RUN:    -DOPT_CC1=-coverage-notes-file \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT
// Check to make sure our '-fsanitize=address' exception isn't triggered by a
// different option
// RUN: %clangxx -fsycl -fprofile-instr-generate=address -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fprofile-instr-generate=address \
// RUN:    -DOPT_CC1=-fprofile-instrument=clang \
// RUN:    -check-prefixes=UNSUPPORTED_OPT_DIAG,UNSUPPORTED_OPT

// CHECK: ignoring '[[OPT]]' option as it is not currently supported for target '[[ARCH]]{{.*}}';only supported for host compilation [-Woption-ignored]
// CHECK-NOT: clang{{.*}} "-fsycl-is-device"{{.*}} "[[OPT]]{{.*}}"
// CHECK: clang{{.*}} "-fsycl-is-host"{{.*}} "[[OPT]]{{.*}}"

// UNSUPPORTED_OPT_DIAG: ignoring '[[OPT]]' option as it is not currently supported for target '[[ARCH]]{{.*}}';only supported for host compilation [-Woption-ignored]
// UNSUPPORTED_OPT-NOT: clang{{.*}} "-fsycl-is-device"{{.*}} "[[OPT_CC1]]{{.*}}"
// UNSUPPORTED_OPT: clang{{.*}} "-fsycl-is-host"{{.*}} "[[OPT_CC1]]{{.*}}"

// FPGA support has been removed, usage of any FPGA specific options and any
// options that have FPGA specific arguments should emit a specific error
// diagnostic.
// RUN: not %clangxx -fintelfpga -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix UNSUPPORTED_FPGA -DBADOPT=-fintelfpga
// RUN: not %clangxx -fsycl -fsycl-targets=spir64_fpga -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix UNSUPPORTED_FPGA -DBADOPT=-fsycl-targets=spir64_fpga
// RUN: not %clangxx -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix UNSUPPORTED_FPGA -DBADOPT=-fsycl-targets=spir64_fpga-unknown-unknown
// RUN: not %clangxx -fsycl -reuse-exe=exe -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix UNSUPPORTED_FPGA -DBADOPT=-reuse-exe=exe
// RUN: not %clangxx -fsycl-help=fpga -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix UNSUPPORTED_FPGA -DBADOPT=-fsycl-help=fpga
// RUN: not %clangxx -fsycl -fintelfpga -Xsycl-target-backend=spir64_fpga "-backend_opts" -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix UNSUPPORTED_FPGA -DBADOPT=-Xsycl-target-backend=spir64_fpga
// RUN: not %clangxx -fsycl -fsycl-link=early -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix UNSUPPORTED_FPGA -DBADOPT=-fsycl-link=early
// RUN: not %clangxx -fsycl -fsycl-link=image -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix UNSUPPORTED_FPGA -DBADOPT=-fsycl-link=image
// UNSUPPORTED_FPGA: option '[[BADOPT]]' is not supported and has been removed from the compiler. Please see the compiler documentation for more details
