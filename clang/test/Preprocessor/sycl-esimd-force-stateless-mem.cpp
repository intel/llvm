/// This test checks that the macro __ESIMD_FORCE_STATELESS_MEM is automatically
/// defined only if the option -fsycl-esimd-force-stateless-mem is used.

// RUN: %clang_cc1 %s -fsycl-is-device -fsycl-esimd-force-stateless-mem -E -dM | FileCheck --check-prefix=CHECK-OPT %s

// RUN: %clang_cc1 %s -E -dM | FileCheck --check-prefix=CHECK-NOOPT %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-NOOPT %s
// RUN: %clang_cc1 %s -fsycl-is-host -E -dM | FileCheck --check-prefix=CHECK-NOOPT %s

// CHECK-OPT:#define __ESIMD_FORCE_STATELESS_MEM 1
// CHECK-NOOPT-NOT:#define __ESIMD_FORCE_STATELESS_MEM 1
