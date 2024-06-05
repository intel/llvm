// Verify the usage of -foffload-lto with SYCL.

// Verify we error when using the old offload driver.
// RUN: not %clangxx -fsycl -foffload-lto=thin %s -### 2>&1 | FileCheck -check-prefix=CHECK_ERROR %s
// CHECK_ERROR: unsupported option '-foffload-lto=thin' for target 'spir64-unknown-unknown'

// Verify there's no error and we see the expected cc1 flags with the new offload driver.
// RUN: %clangxx -fsycl --offload-new-driver -foffload-lto=thin %s -### 2>&1 | FileCheck -check-prefix=CHECK_SUPPORTED %s
// CHECK_SUPPORTED: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" {{.*}} "-flto=thin" "-flto-unit"
