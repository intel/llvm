/// Check that no device traits macros are defined if sycl is disabled: 
// RUN:   %clang -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DISABLED %s
// CHECK-DISABLED-NOT: "{{.*}}SYCL_ANY_DEVICE_HAS{{.*}}"
// CHECK-DISABLED-NOT: "{{.*}}SYCL_ALL_DEVICES_HAVE{{.*}}"

/// Check device traits macros are defined if sycl is enabled: 
/// In this case, where no specific sycl targets are passed, the sycl
/// targets are spir64 and the host target (e.g. x86_64). We expect two
/// occurrences of the macro definition, one for host and one for device.
// RUN:   %clang -fsycl -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ENABLED %s
// CHECK-ENABLED-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"

/// Check device traits macros are defined if sycl is enabled: 
/// In this case the sycl targets are spir64, spir64_gen and the host
/// target (e.g. x86_64). We expect three occurrences of the macro
/// definition, one for host and one for each of the two devices.
// RUN:   %clang -fsycl -fsycl-targets=spir64,spir64_gen -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-TARGETS %s
// CHECK-SYCL-TARGETS-COUNT-3: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
