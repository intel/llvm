
/// Verify that the driver option is translated to corresponding options
/// to host
// RUN: %clang -### -fsycl -fno-sycl-esimd-build-host-code \
// RUN: %s 2>&1 | FileCheck -check-prefix=CHECK-PASS-TO-COMPS %s
// CHECK-PASS-TO-COMPS: clang
// CHECK-PASS-TO-COMPS: clang{{.*}} "-fsycl-is-host" {{.*}}"-fno-sycl-esimd-build-host-code"
// CHECK-PASS-TO-COMPS-NOT: "-fno-sycl-esimd-build-host-code"
// CHECK-PASS-TO-COMPS: sycl-post-link{{.*}}

/// Verify that removing host code is not enabled by default
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck -implicit-check-not "-fno-sycl-esimd-build-host-code" %s
