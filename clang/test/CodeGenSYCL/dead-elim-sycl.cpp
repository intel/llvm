// RUN: %clang -cc1 -triple spir64-unknown-unknown  -fsycl-is-device  -emit-llvm-bc  -O1 -fno-legacy-pass-manager -fdebug-pass-manager -fenable-sycl-dae -o /dev/null -x c++ < %s 2>&1 | FileCheck %s
// RUN: %clang -cc1 -triple spir64-unknown-unknown  -fsycl-is-device  -emit-llvm-bc  -O0 -fno-legacy-pass-manager -fdebug-pass-manager  -o /dev/null -x c++ < %s 2>&1 | FileCheck %s --check-prefix DISABLE
// RUN: %clang -cc1 -triple spir64-unknown-unknown  -fsycl-is-device  -emit-llvm-bc  -O1 -flegacy-pass-manager -mllvm -debug-pass=Structure -fenable-sycl-dae -o /dev/null -x c++ < %s 2>&1 | FileCheck %s --check-prefix OLDPM
// RUN: %clang -cc1 -triple spir64-unknown-unknown  -fsycl-is-device  -emit-llvm-bc  -O1 -flegacy-pass-manager -mllvm -debug-pass=Structure -o /dev/null -x c++ < %s 2>&1 | FileCheck %s --check-prefix DISABLE

// Verify that Dead Arguments Elimination for SYCL kernels is/is not added to the PM.

// CHECK: Running pass: DeadArgumentEliminationSYCLPass on [module]
// OLDPM: Dead Argument Elimination for SYCL kernels
// DISABLE-NOT: DeadArgumentEliminationSYCLPass
// DISABLE-NOT: Dead Argument Elimination for SYCL kernels
