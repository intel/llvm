/// Check that SPIR ITT instrumentation is disabled by default:
// RUN: %clang -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fsycl-instrument-device-code"

/// Check if "fsycl_instrument_device_code" is passed to -cc1:
// RUN: %clang -### -fsycl-instrument-device-code %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-ENABLED %s
// CHECK-ENABLED: "-cc1"{{.*}} "-fsycl-instrument-device-code"

/// Check if "fsycl_instrument_device_code" usage with a non-spirv target
/// results in an error.
// RUN: %clang -### -fsycl-instrument-device-code --target=x86 %s 2>&1
// expected-error{{unsupported option '-fsycl-instrument-device-code' for target 'x86_64-unknown-linux-gnu'}}
