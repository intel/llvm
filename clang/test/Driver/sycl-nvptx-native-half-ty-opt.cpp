/// Check that -fnative-half-type is added automatically for SM versions
/// greater or equal to 53.
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_53 %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-53 %s
// CHECK-53: clang{{.*}} "-triple" "nvptx64-nvidia-cuda"{{.*}}"-fnative-half-type"
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_70 %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-70 %s
// CHECK-70: clang{{.*}} "-triple" "nvptx64-nvidia-cuda"{{.*}}"-fnative-half-type"

/// As the default is set to SM_50, make sure that the option is not added.
// RUN: not %clangxx -fsycl -nocudalib -fsycl -fsycl-targets=nvidia64-nvidia-cuda %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: {{.*}}"-fnative-half-type"

/// SM < 53
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_50 %s -### 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-50 %s
// CHECK-50-NOT: {{.*}}"-fnative-half-type"
