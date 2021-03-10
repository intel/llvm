/// Check that SYCL ITT instrumentation is disabled by default:
// RUN: %clang -fsycl -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-add-instrumentation-calls"

/// Check "fsycl_device_code_add_instrumentation_calls" is passed to sycl post
/// link tool:
// RUN: %clang -fsycl -### -fsycl-instrument-device-code %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-ENABLED %s
// CHECK-ENABLED: "-add-instrumentation-calls"
