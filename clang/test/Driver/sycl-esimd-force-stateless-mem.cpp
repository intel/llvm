
/// Verify that the driver option is translated to corresponding options
/// to device compilation and sycl-post-link.
// RUN: %clang -### -fsycl -fsycl-esimd-force-stateless-mem \
// RUN: %s 2>&1 | FileCheck -check-prefix=CHECK-PASS-TO-COMPS %s
// CHECK-PASS-TO-COMPS: clang{{.*}} "-fsycl-esimd-force-stateless-mem"
// CHECK-PASS-TO-COMPS: sycl-post-link{{.*}} "-lower-esimd-force-stateless-mem"
// CHECK-PASS-TO-COMPS-NOT: clang{{.*}} "-fsycl-is-host" {{.*}}"-fsycl-esimd-force-stateless-mem"
// CHECK-PASS-TO-COMPS-NOT: clang{{.*}} "-fsycl-esimd-force-stateless-mem" {{.*}}"-fsycl-is-host"

/// Verify that stateless memory accesses mapping is not enforced by default
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: clang{{.*}} "-fsycl-esimd-force-stateless-mem"
// CHECK-DEFAULT-NOT: sycl-post-link{{.*}} "-lower-esimd-force-stateless-mem"
