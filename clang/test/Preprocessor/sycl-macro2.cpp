// RUN: %clang -c %s -fsycl -E -dM -o - | FileCheck --check-prefix=CHECK-MSVC %s
// REQUIRES: system-windows
// CHECK-MSVC-NOT: __GNUC__
// CHECK-MSVC-NOT: __STDC__
