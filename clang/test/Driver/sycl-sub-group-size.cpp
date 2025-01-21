// Ensure that by default the -fsycl-default-sub-group-size doesn't get passed
// to the cc1 invocation.
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT-NOT: "fsycl-default-sub-group-size"

// The next two tests make sure that the -fsycl-default-sub-group-size command
// line arguments get properly passed unaltered to the cc1 invocation.
// RUN: %clang -### -fsycl -fsycl-default-sub-group-size=primary %s 2>&1 | FileCheck %s -check-prefix=PRIM
// PRIM: "-fsycl-default-sub-group-size" "primary"

// RUN: %clang -### -fsycl -fsycl-default-sub-group-size=10 %s 2>&1 | FileCheck %s -check-prefix=TEN
// TEN: "-fsycl-default-sub-group-size" "10"
