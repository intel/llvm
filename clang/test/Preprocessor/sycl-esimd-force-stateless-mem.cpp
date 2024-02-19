/// This test checks that the macro __ESIMD_FORCE_STATELESS_MEM is automatically
/// defined by default with -fsycl-is-device or -fsycl-is-host.

// RUN: %clang_cc1 %s -fsycl-is-device -fsycl-esimd-force-stateless-mem -E -dM | FileCheck --check-prefix=CHECK-OPT %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-OPT %s
// RUN: %clang_cc1 %s -fsycl-is-host -E -dM | FileCheck --check-prefix=CHECK-OPT %s

// RUN: %clang_cc1 %s -E -dM | FileCheck --check-prefix=CHECK-NOOPT %s
// RUN: %clang_cc1 %s -fsycl-is-device -fno-sycl-esimd-force-stateless-mem -E -dM | FileCheck --check-prefix=CHECK-NOOPT %s
// RUN: %clang_cc1 %s -fsycl-is-host -fno-sycl-esimd-force-stateless-mem -E -dM | FileCheck --check-prefix=CHECK-NOOPT %s

// CHECK-OPT:#define __ESIMD_FORCE_STATELESS_MEM 1
// CHECK-NOOPT-NOT:#define __ESIMD_FORCE_STATELESS_MEM 1
