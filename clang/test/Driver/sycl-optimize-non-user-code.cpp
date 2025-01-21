// RUN: %clangxx -fsycl -O0 -### %s 2>&1 | FileCheck %s -check-prefix=NO-OPT-CHECK
// NO-OPT-CHECK-NOT: fsycl-optimize-non-user-code

// RUN: %clangxx -fsycl -O0 -fsycl-optimize-non-user-code -### %s 2>&1 | FileCheck %s -check-prefix=OPT-CHECK
// OPT-CHECK: fsycl-optimize-non-user-code

// RUN: not %clangxx -fsycl -O1 -fsycl-optimize-non-user-code %s 2>&1  | FileCheck %s -check-prefix=CHECK-ERROR
// RUN: not %clangxx -fsycl -fsycl-optimize-non-user-code %s 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR
// CHECK-ERROR: error: -fsycl-optimize-non-user-code option can be used only in conjunction with -O0

// Check cases for Microsoft Windows Driver.
// RUN: %clang_cl -fsycl -Od -### %s 2>&1 | FileCheck %s -check-prefix=NO-OPT-WIN-CHECK
// NO-OPT-WIN-CHECK-NOT: fsycl-optimize-non-user-code

// RUN: %clang_cl -fsycl -Od -fsycl-optimize-non-user-code -### %s 2>&1 | FileCheck %s -check-prefix=OPT-WIN-CHECK
// OPT-WIN-CHECK: fsycl-optimize-non-user-code

// RUN: not %clang_cl -fsycl -O1 -fsycl-optimize-non-user-code %s 2>&1  | FileCheck %s -check-prefix=CHECK-WIN-ERROR
// RUN: not %clang_cl -fsycl -fsycl-optimize-non-user-code %s 2>&1 | FileCheck %s -check-prefix=CHECK-WIN-ERROR
// CHECK-WIN-ERROR: error: -fsycl-optimize-non-user-code option can be used only in conjunction with -Od
