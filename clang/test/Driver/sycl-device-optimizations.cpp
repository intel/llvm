



/// Check that optimizations for sycl device are enabled with -g and O2 passed:

// RUN:   %clang_cl -### -fsycl -O2 -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-G-O2 %s
// CHECK-G-O2: clang{{.*}} "-fsycl-is-device{{.*}}" "-O3"
// CHECK-G-O2: sycl-post-link{{.*}} "-O3"
// CHECK-G-O2-NOT: "-O0"





