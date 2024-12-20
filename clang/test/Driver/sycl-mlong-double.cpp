/// -mlong-double-64 is valid for host and device for -fsycl
// RUN: %clangxx -c -fsycl -mlong-double-64 -target x86_64-unknown-linux-gnu %s -### 2>&1 | FileCheck %s
// CHECK: clang{{.*}} "-triple" "spir64-unknown-unknown"
// CHECK-SAME:  "-mlong-double-64"
// CHECK: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME:  "-mlong-double-64"

/// -mlong-double-128 and -mlong-double-80 are not supported for spir64, or SYCL GPU targets.
// RUN: not %clangxx -c -fsycl -mlong-double-128 -target x86_64-unknown-linux-gnu %s -### 2>&1 | FileCheck --check-prefix=CHECK-128 %s
// CHECK-128: error: unsupported option '-mlong-double-128' for target 'spir64-unknown-unknown'
// CHECK-128-NOT: clang{{.*}} "-triple" "-spir64-unknown-unknown" {{.*}} "-mlong-double-128"

// RUN: not %clangxx -c -fsycl -mlong-double-80 -target x86_64-unknown-linux-gnu %s -### 2>&1 | FileCheck --check-prefix=CHECK-80 %s
// CHECK-80: error: unsupported option '-mlong-double-80' for target 'spir64-unknown-unknown'
// CHECK-80-NOT: clang{{.*}} "-triple" "-spir64-unknown-unknown" {{.*}} "-mlong-double-80"

// RUN: not %clangxx -c -fsycl -mlong-double-128 -target amd_gpu_gfx1031 %s -### 2>&1 | FileCheck --check-prefix=CHECK-128-AMD %s
// CHECK-128-AMD: error: unsupported option '-mlong-double-128' for target 'amd_gpu_gfx1031'
// CHECK-128-AMD-NOT: clang{{.*}} "-triple" "-amd_gpu_gfx1031" {{.*}} "-mlong-double-128"

// RUN: not %clangxx -c -fsycl -mlong-double-80 -target amd_gpu_gfx1031 %s -### 2>&1 | FileCheck --check-prefix=CHECK-80-AMD %s
// CHECK-80-AMD: error: unsupported option '-mlong-double-80' for target 'amd_gpu_gfx1031'
// CHECK-80-AMD-NOT: clang{{.*}} "-triple" "-amd_gpu_gfx1031" {{.*}} "-mlong-double-80"

// RUN: not %clangxx -c -fsycl -mlong-double-128 -target nvptx64-nvidia-cuda %s -### 2>&1 | FileCheck --check-prefix=CHECK-128-NVPTX %s
// CHECK-128-NVPTX: error: unsupported option '-mlong-double-128' for target 'nvptx64-nvidia-cuda'
// CHECK-128-NVPTX-NOT: clang{{.*}} "-triple" "-nvptx64-nvidia-cuda" {{.*}} "-mlong-double-128"

// RUN: not %clangxx -c -fsycl -mlong-double-80 -target nvptx64-nvidia-cuda %s -### 2>&1 | FileCheck --check-prefix=CHECK-80-NVPTX %s
// CHECK-80-NVPTX: error: unsupported option '-mlong-double-80' for target 'nvptx64-nvidia-cuda'
// CHECK-80-NVPTX-NOT: clang{{.*}} "-triple" "-nvptx64-nvidia-cuda" {{.*}} "-mlong-double-80"
