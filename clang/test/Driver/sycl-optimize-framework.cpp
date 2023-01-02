// RUN: %clangxx -fsycl -O0 -### %s 2>&1 | FileCheck %s -check-prefix=NO-OPT-CHECK
// NO-OPT-CHECK-NOT: fsycl-optimize-framework

// RUN: %clangxx -fsycl -O0 -fsycl-optimize-framework -### %s 2>&1 | FileCheck %s -check-prefix=OPT-CHECK
// OPT-CHECK: fsycl-optimize-framework

// RUN: not %clangxx -fsycl -O1 -fsycl-optimize-framework %s 2>&1  | FileCheck %s -check-prefix=CHECK-ERROR
// RUN: not %clangxx -fsycl -fsycl-optimize-framework %s 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR
// CHECK-ERROR: error: -fsycl-optimize-framework option can be used only in conjunction with -O0 option

// Check cases for Microsoft Windows Driver.
// RUN: %clang_cl -fsycl -Od -### %s 2>&1 | FileCheck %s -check-prefix=NO-OPT-WIN-CHECK
// NO-OPT-WIN-CHECK-NOT: fsycl-optimize-framework

// RUN: %clang_cl -fsycl -Od -fsycl-optimize-framework -### %s 2>&1 | FileCheck %s -check-prefix=OPT-WIN-CHECK
// OPT-WIN-CHECK: fsycl-optimize-framework

// RUN: not %clang_cl -fsycl -O1 -fsycl-optimize-framework %s 2>&1  | FileCheck %s -check-prefix=CHECK-WIN-ERROR
// RUN: not %clang_cl -fsycl -fsycl-optimize-framework %s 2>&1 | FileCheck %s -check-prefix=CHECK-WIN-ERROR
// CHECK-WIN-ERROR: error: -fsycl-optimize-framework option can be used only in conjunction with -Od option
