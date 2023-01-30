/// -mlong-double-64 is valid for host and device for -fsycl
// RUN: %clangxx -c -fsycl -mlong-double-64 -target x86_64-unknown-linux-gnu %s -### 2>&1 | FileCheck %s
// CHECK: clang{{.*}} "-triple" "spir64-unknown-unknown"
// CHECK-SAME:  "-mlong-double-64"
// CHECK: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME:  "-mlong-double-64"

/// -mlong-double-128 and -mlong-double-80 are not supported for spir64.
// RUN: %clangxx -c -fsycl -mlong-double-128 -target x86_64-unknown-linux-gnu %s -### 2>&1 | FileCheck --check-prefix=CHECK-128 %s
// CHECK-128: error: unsupported option '-mlong-double-128' for target 'spir64-unknown-unknown'
// CHECK-128-NOT: clang{{.*}} "-triple" "-spir64-unknown-unknown" {{.*}} "-mlong-double-128"

// RUN: %clangxx -c -fsycl -mlong-double-80 -target x86_64-unknown-linux-gnu %s -### 2>&1 | FileCheck --check-prefix=CHECK-80 %s
// CHECK-80: error: unsupported option '-mlong-double-80' for target 'spir64-unknown-unknown'
// CHECK-80-NOT: clang{{.*}} "-triple" "-spir64-unknown-unknown" {{.*}} "-mlong-double-80"
