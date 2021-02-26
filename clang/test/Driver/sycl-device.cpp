/// Check that compiling for sycl device is disabled by default:
// RUN:   %clang -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fsycl-is-device"

/// Check "-fsycl-is-device" is passed when compiling for device:
// RUN:   %clang -### -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-DEV %s
// CHECK-SYCL-DEV: "-fsycl-is-device"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"

/// Check that "-Wno-sycl-strict" is set on compiler invocation with "-fsycl"
/// or "-fsycl-device-only" or both:
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// RUN:   %clang -### -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// RUN:   %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// CHECK-SYCL-NO_STRICT: clang{{.*}} "-Wno-sycl-strict"

/// Check that -sycl-std=2017 is set if no std version is provided by user
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-STD_VERSION %s
// CHECK-SYCL-STD_VERSION: clang{{.*}} "-sycl-std=2020"

/// Check that -aux-triple is set correctly
// RUN:   %clang -### -fsycl -target aarch64-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-AUX-TRIPLE %s
// TODO: %clang -### -fsycl -fsycl-device-only -target aarch64-linux-gnu
// CHECK-SYCL-AUX-TRIPLE: clang{{.*}} "-aux-triple" "aarch64-unknown-linux-gnu"

/// Verify output files are properly specified given -o
// RUN: %clang -### -fsycl -fsycl-device-only -o dummy.out %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-OUTPUT-FILE %s
// RUN: %clang_cl -### -fsycl -fsycl-device-only -o dummy.out %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-OUTPUT-FILE %s
// CHECK-OUTPUT-FILE: clang{{.*}} "-o" "dummy.out"
