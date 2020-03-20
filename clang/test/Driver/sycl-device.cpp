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
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT1 %s
// CHECK-SYCL-NO_STRICT1: "-Wno-sycl-strict"

// RUN:   %clang -### -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT2 %s
// CHECK-SYCL-NO_STRICT2: "-Wno-sycl-strict"

// RUN:   %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT3 %s
// CHECK-SYCL-NO_STRICT3: "-Wno-sycl-strict"
