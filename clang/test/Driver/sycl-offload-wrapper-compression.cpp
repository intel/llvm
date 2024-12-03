///
/// Check if '--offload-compress' and '--offload-compression-level' CLI
/// options are passed to the clang-offload-wrapper.
///

// RUN: %clangxx -### -fsycl --offload-compress --offload-compression-level=3 %s 2>&1 | FileCheck %s --check-prefix=CHECK-COMPRESS
// CHECK-COMPRESS: {{.*}}clang-offload-wrapper{{.*}}"-offload-compress"{{.*}}"-offload-compression-level=3"{{.*}}

// Make sure that the compression options are not passed when --offload-compress is not set.
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-COMPRESS
// RUN: %clangxx -### -fsycl --offload-compression-level=3 %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-COMPRESS

// For SYCL offloading to HIP, make sure we don't pass '--compress' to offload-bundler.
// RUN: %clangxx -### -fsycl --offload-compress -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1031 -fsycl-targets=amdgcn-amd-amdhsa -fno-sycl-libspirv -nogpulib %s &> %t.driver
// RUN: FileCheck %s --check-prefix=CHECK-NO-COMPRESS-BUNDLER --input-file=%t.driver

// CHECK-NO-COMPRESS-NOT: {{.*}}clang-offload-wrapper{{.*}}"-offload-compress"{{.*}}
// CHECK-NO-COMPRESS-NOT: {{.*}}clang-offload-wrapper{{.*}}"-offload-compression-level=3"{{.*}}

// CHECK-NO-COMPRESS-BUNDLER-NOT: {{.*}}clang-offload-bundler{{.*}}"-compress"{{.*}}
