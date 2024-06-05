// Test checks -fsycl-add-default-spec-consts-image flag.

// Check usages when warning should be issued.
// RUN: %clang -### -fsycl -fsycl-add-default-spec-consts-image 2>&1 %s | FileCheck %s -check-prefix=CHECK-NON-AOT
// RUN: %clang_cl -### -fsycl -fsycl-add-default-spec-consts-image 2>&1 %s | FileCheck %s -check-prefix=CHECK-NON-AOT
// RUN: %clang -### -fsycl -fsycl-add-default-spec-consts-image -fsycl-targets=spir64 2>&1 %s | FileCheck %s -check-prefix=CHECK-NON-AOT
// CHECK-NON-AOT: warning: -fsycl-add-default-spec-consts-image flag has an effect only in Ahead of Time Compilation mode (AOT)

// Check that non-AOT target doesn't add command line option into sycl-post-link invocation
// CHECK-NON-AOT-NOT: {{.*}}sycl-post-link{{.*}} "-generate-device-image-default-spec-consts"

// Check that no warnings are issued in correct cases and "-generate-device-image-default-spec-consts" is passed to sycl-post-link
// RUN: %clang -### -fsycl -fsycl-add-default-spec-consts-image -fsycl-targets=spir64_gen 2>&1  %s | FileCheck %s -check-prefix=CHECK-AOT
// RUN: %clang -### -fsycl -fsycl-add-default-spec-consts-image -fsycl-targets=spir64_fpga 2>&1 %s | FileCheck %s -check-prefix=CHECK-AOT
// RUN: %clang -### -fsycl -fsycl-add-default-spec-consts-image -fsycl-targets=spir64_x86_64 2>&1 %s | FileCheck %s -check-prefix=CHECK-AOT
// RUN: %clang -### -fsycl -fsycl-add-default-spec-consts-image -fsycl-targets=intel_gpu_pvc 2>&1 %s | FileCheck %s -check-prefix=CHECK-AOT
// RUN: %clang -### -fsycl -fsycl-add-default-spec-consts-image -fsycl-targets=nvidia_gpu_sm_90 -nocudalib 2>&1 %s | FileCheck %s -check-prefix=CHECK-AOT
// RUN: %clang -### -fsycl -fsycl-add-default-spec-consts-image -fsycl-targets=amd_gpu_gfx1034 -fno-sycl-libspirv -nogpulib  2>&1 %s | FileCheck %s -check-prefix=CHECK-AOT
// CHECK-AOT-NOT: warning: -fsycl-add-default-spec-consts-image flag has an effect only in Ahead of Time Compilation mode (AOT)
// CHECK-AOT: {{.*}}sycl-post-link{{.*}} "-generate-device-image-default-spec-consts"


// RUN: %clang -### -fsycl -fno-sycl-add-default-spec-consts-image -fsycl-targets=spir64_gen 2>&1  %s | FileCheck %s -check-prefix=CHECK-NO-ADD
// RUN: %clang_cl -### -fsycl -fno-sycl-add-default-spec-consts-image -fsycl-targets=spir64_gen 2>&1  %s | FileCheck %s -check-prefix=CHECK-NO-ADD
// CHECK-NO-ADD-NOT: {{.*}}sycl-post-link{{.*}} "-generate-device-image-default-spec-consts"
