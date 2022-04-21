/// Check that -fopenmp-simd is NOT passed to the device compilation when
/// used together with -fsycl.

// REQUIRES: clang-driver

// RUN: %clang -c -fsycl -fopenmp-simd -### %s 2>&1 | FileCheck %s
// RUN: %clang -c -fopenmp-simd -fopenmp-version=50 -### %s 2>&1 | FileCheck %s --check-prefix=UNCHANGED

// CHECK-NOT: "-triple" "spir64"{{.*}} "-fsycl-is-device"{{.*}} "-target=spir64"{{.*}} "-fopenmp-simd"{{.*}} "-fopenmp-version=50"
// CHECK: "-triple"{{.*}} "-fsycl-is-host"{{.*}} "-fopenmp-simd"{{.*}}

// UNCHANGED: "-triple"{{.*}} "-fopenmp-simd"{{.*}} "-fopenmp-version=50"{{.*}}

void foo(long double) {}
