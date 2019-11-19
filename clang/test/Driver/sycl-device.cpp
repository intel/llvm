/// Check that compiling for sycl device is disabled by default:
// RUN:   %clang -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fsycl-is-device"

/// Check "-fsycl-is-device" is passed when compiling for device:
// RUN:   %clang -### -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-DEV %s
// CHECK-SYCL-DEV: "-fsycl-is-device"
