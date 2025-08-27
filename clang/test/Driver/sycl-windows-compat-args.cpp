// RUN: %clang -### -fsycl-device-only -target x86_64-unknown-windows-msvc -c %s 2>&1 | FileCheck %s
// CHECK: -fms-compatibility
// CHECK: -fdelayed-template-parsing
// expected-no-diagnostics

// This test checks that the driver passes -fdelayed-template-parsing
// and -fms-compatibility on Windows for SYCL, both of which are needed for
// successful compilation of C++ code constructs used in header files.

// Check that the compatibility version is set according to triple for device
// RUN: %clang_cl -### -fsycl-device-only --target=x86_64-unknown-windows-msvc19.29.0 -c %s 2>&1 | FileCheck %s -check-prefix=COMPAT_VERSION
// COMPAT_VERSION: -fms-compatibility-version=19.29.0
