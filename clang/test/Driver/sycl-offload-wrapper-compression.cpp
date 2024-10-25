///
/// Check if '--offload-compress' and '--offload-compression-level' CLI
/// options are passed to the clang-offload-wrapper.
///

// RUN: %clangxx -### -fsycl --offload-compress --offload-compression-level=3 %s 2>&1 | FileCheck %s --check-prefix=CHECK-COMPRESS
// CHECK-COMPRESS: {{.*}}clang-offload-wrapper{{.*}}"-offload-compress"{{.*}}"-offload-compression-level=3"{{.*}}

// Make sure that the compression options are not passed when --offload-compress is not set.
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-COMPRESS
// RUN: %clangxx -### -fsycl --offload-compression-level=3 %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-COMPRESS

// CHECK-NO-COMPRESS-NOT: {{.*}}clang-offload-wrapper{{.*}}"-offload-compress"{{.*}}
// CHECK-NO-COMPRESS-NOT: {{.*}}clang-offload-wrapper{{.*}}"-offload-compression-level=3"{{.*}}
