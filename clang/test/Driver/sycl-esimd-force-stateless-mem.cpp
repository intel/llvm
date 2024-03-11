
/// Verify that the driver option is translated to corresponding options
/// to host/device compilation and sycl-post-link.

// Case1: Check that the enforcing is turned on by default.
// Actually, only sycl-post-link gets an additional flag in this case.
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: clang{{.*}} "sycl-esimd-force-stateless-mem"
// CHECK-DEFAULT-NOT: clang{{.*}} "-fsycl-is-host" {{.*}}"sycl-esimd-force-stateless-mem"
// CHECK-DEFAULT-NOT: sycl-post-link{{.*}} "-lower-esimd-force-stateless-mem"

// Case2: Check that -fno-sycl-esimd-force-stateless-mem is handled correctly -
// i.e. sycl-post-link gets nothing and clang gets corresponding -fno... option.
// RUN: %clang -### -fsycl -fno-sycl-esimd-force-stateless-mem %s 2>&1 | FileCheck -check-prefix=CHECK-NEG %s
// CHECK-NEG: clang{{.*}} "-fno-sycl-esimd-force-stateless-mem"
// CHECK-NEG: sycl-post-link{{.*}} "-lower-esimd-force-stateless-mem=false"
// CHECK-NEG-NOT: clang{{.*}} "-fsycl-is-host" {{.*}}"sycl-esimd-force-stateless-mem"
