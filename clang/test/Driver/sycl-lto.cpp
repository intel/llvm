// Verify the usage of -foffload-lto with SYCL.
// RUN: touch %t.cpp
// RUN: not %clangxx -fsycl -foffload-lto=thin %t.cpp -### 2>&1 | FileCheck -check-prefix=CHECK_ERROR %s
//
// RUN: %clangxx -fsycl --offload-new-driver -foffload-lto=thin %t.cpp -### 2>&1 | FileCheck -check-prefix=CHECK_SUPPORTED %s

// CHECK_ERROR: unsupported option '-foffload-lto=thin' for target 'spir64-unknown-unknown'

// CHECK_SUPPORTED: clang-19{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" {{.*}} "-flto=thin" "-flto-unit"
