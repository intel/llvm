/// Check that explicit SIMD extension is disabled by default:
// RUN: %clang -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fsycl-explicit-simd"

/// Check "-fsycl-explicit-simd" is passed when compiling for device and host:
// RUN: %clang -### -fsycl -fsycl-explicit-simd %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-SYCL-ESIMD %s
// CHECK-SYCL-ESIMD: "-cc1"{{.*}} "-fsycl-explicit-simd"{{.*}}
// CHECK-SYCL-ESIMD: "-cc1"{{.*}} "-fsycl-explicit-simd"{{.*}}
