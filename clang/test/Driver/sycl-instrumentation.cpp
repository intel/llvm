/// Check that SYCL ITT instrumentation is disabled by default:
// RUN: %clang -fsycl -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "fsycl-instrument-device-code"

/// Check if "fsycl_instrument_device_code" is passed to sycl post
/// link tool:
// RUN: %clang -fsycl -### -fsycl-instrument-device-code %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-ENABLED %s
// CHECK-ENABLED: "fsycl-instrument-device-code"
