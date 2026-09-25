// Verify the usage of -foffload-lto with SYCL.

// Verify we error when using the old offload driver.
// RUN: not %clangxx -fsycl -foffload-lto=thin %s -### 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s
// CHECK-ERROR: unsupported option '-foffload-lto=thin' for target 'spir64-unknown-unknown'

// Verify we error when using the new offload driver but with device code split set to off.
// RUN: not %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -foffload-lto=thin -fsycl-device-code-split=off %s -### 2>&1 | FileCheck -check-prefix=CHECK-SPLIT-ERROR %s
// CHECK-SPLIT-ERROR: '-fsycl-device-code-split=off' is not supported when '-foffload-lto=thin' is set with '-fsycl'

// Verify there's no error and we see the expected cc1 flags and tool invocations with the new offload driver.
// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -foffload-lto=thin %s -### 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SUPPORTED -implicit-check-not=-emit-only-kernels-as-entry-points %s
// CHECK-SUPPORTED: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" {{.*}} "-flto=thin" "-flto-unit"
// CHECK-SUPPORTED: sycl-post-link
// CHECK-SUPPORTED-NOT: -properties
// CHECK-SUPPORTED-NEXT: file-table-tform{{.*}}
// CHECK-SUPPORTED-NEXT: llvm-foreach{{.*}} "--" {{.*}}clang{{.*}} "-fsycl-is-device"{{.*}} "-flto=thin" "-flto-unit"
// CHECK-SUPPORTED-NEXT: file-table-tform{{.*}}
// CHECK-SUPPORTED-NEXT: llvm-offload-binary{{.*}} "-o" "{{.*}}" "--image=file=@{{.*}}"
// CHECK-SUPPORTED: clang-linker-wrapper{{.*}} "-sycl-thin-lto"

// Verify that for AOT (ocloc/opencl-aot) targets the LTO mode is still used
// for the actual device compile, but is not forwarded to clang-linker-wrapper
// via --device-compiler=/--device-linker=, since ocloc/opencl-aot do not
// accept -flto= as a raw command line option.
// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -fsycl-targets=spir64_gen,spir64_x86_64 -foffload-lto=thin %s -### 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-AOT-THIN %s
// CHECK-AOT-THIN: clang{{.*}} "-cc1" "-triple" "spir64_gen-unknown-unknown" {{.*}} "-flto=thin" "-flto-unit"
// CHECK-AOT-THIN: clang{{.*}} "-cc1" "-triple" "spir64_x86_64-unknown-unknown" {{.*}} "-flto=thin" "-flto-unit"
// CHECK-AOT-THIN: clang-linker-wrapper
// CHECK-AOT-THIN-NOT: "--device-compiler={{.*}}=-flto=
// CHECK-AOT-THIN-NOT: "--device-linker={{.*}}=-flto=

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -fsycl-targets=spir64_gen -foffload-lto=full %s -### 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-AOT-FULL %s
// CHECK-AOT-FULL: clang{{.*}} "-cc1" "-triple" "spir64_gen-unknown-unknown" {{.*}} "-flto=full"
// CHECK-AOT-FULL: clang-linker-wrapper
// CHECK-AOT-FULL-NOT: "--device-compiler={{.*}}=-flto=
// CHECK-AOT-FULL-NOT: "--device-linker={{.*}}=-flto=
