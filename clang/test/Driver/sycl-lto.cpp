// Verify the usage of -foffload-lto with SYCL.

// Verify we error when using the old offload driver.
// RUN: not %clangxx -fsycl -foffload-lto=thin %s -### 2>&1 | FileCheck -check-prefix=CHECK_ERROR %s
// CHECK_ERROR: unsupported option '-foffload-lto=thin' for target 'spir64-unknown-unknown'

// Verify we error when using the new offload driver but with device code split set to off.
// RUN: not %clangxx -fsycl --offload-new-driver -foffload-lto=thin -fsycl-device-code-split=off %s -### 2>&1 | FileCheck -check-prefix=CHECK_SPLIT_ERROR %s
// CHECK_SPLIT_ERROR: '-fsycl-device-code-split=off' is not supported when '-foffload-lto=thin' is set with '-fsycl'

// Verify there's no error and we see the expected cc1 flags and tool invocations with the new offload driver.
// RUN: %clangxx -fsycl --offload-new-driver -foffload-lto=thin %s -### 2>&1 | FileCheck -check-prefix=CHECK_SUPPORTED %s
// CHECK_SUPPORTED: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" {{.*}} "-flto=thin" "-flto-unit"
// CHECK_SUPPORTED: sycl-post-link{{.*}}
// CHECK_SUPPORTED-NOT: -properties
// CHECK_SUPPORTED-NEXT: file-table-tform{{.*}}
// CHECK_SUPPORTED-NEXT: llvm-foreach{{.*}} "--" {{.*}}clang{{.*}} "-fsycl-is-device"{{.*}} "-flto=thin" "-flto-unit"
// CHECK_SUPPORTED-NEXT: file-table-tform{{.*}}
// CHECK_SUPPORTED-NEXT: clang-offload-packager{{.*}} "-o" "{{.*}}" "--image=file=@{{.*}}"
// CHECK_SUPPORTED: clang-linker-wrapper{{.*}} "-sycl-thin-lto"
