// Verify the usage of -foffload-lto with SYCL.
// RUN: not %clangxx -fsycl -foffload-lto=thin %s -### 2>&1 | FileCheck -check-prefix=CHECK_ERROR %s
//
// RUN: %clangxx -fsycl --offload-new-driver -foffload-lto=thin %s -### 2>&1 | FileCheck -check-prefix=CHECK_SUPPORTED %s

// CHECK_ERROR: unsupported option '-foffload-lto=thin' for target 'spir64-unknown-unknown'

// CHECK_SUPPORTED: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" {{.*}} "-flto=thin" "-flto-unit"
