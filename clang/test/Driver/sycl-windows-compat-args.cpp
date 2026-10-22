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

// -fdelayed-template-parsing is not enabled by default for C++20 and later,
// matching MSVC and the host compilation.
// RUN: %clang -### -fsycl-device-only -std=c++20 --target=x86_64-unknown-windows-msvc -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NO_DELAYED
// RUN: %clang_cl -### -fsycl-device-only /std:c++20 --target=x86_64-unknown-windows-msvc -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NO_DELAYED
// RUN: %clang_cl -### -fsycl /std:c++20 --target=x86_64-unknown-windows-msvc -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NO_DELAYED

// -fno-delayed-template-parsing is honored for the device compilation.
// RUN: %clang -### -fsycl-device-only -fno-delayed-template-parsing --target=x86_64-unknown-windows-msvc -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NO_DELAYED
// RUN: %clang_cl -### -fsycl -fno-delayed-template-parsing --target=x86_64-unknown-windows-msvc -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NO_DELAYED
// NO_DELAYED-NOT: -fdelayed-template-parsing

// Host and device compilations agree on -fdelayed-template-parsing.
// RUN: %clang_cl -### -fsycl /std:c++17 --target=x86_64-unknown-windows-msvc -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DELAYED_BOTH
// DELAYED_BOTH: clang{{.*}} "-fsycl-is-device"{{.*}} "-fdelayed-template-parsing"
// DELAYED_BOTH: clang{{.*}} "-fsycl-is-host"{{.*}} "-fdelayed-template-parsing"
