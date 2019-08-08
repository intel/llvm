// RUN: %clang -### --sycl -target x86_64-unknown-windows-msvc -c %s 2>&1 | FileCheck %s
// CHECK: -fms-compatibility
// CHECK: -fdelayed-template-parsing
// expected-no-diagnostics

// This test checks that the driver passes -fdelayed-template-parsing
// and -fms-compatibility on Windows for SYCL, both of which are needed for
// successful compilation of C++ code constructs used in header files.
