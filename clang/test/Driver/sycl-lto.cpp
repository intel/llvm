// Verify the usage of -foffload-lto with SYCL.

// DEFINE: %{clang_install_dir} = %/S/Inputs/SYCL/bin

// Verify we error when using the old offload driver.
// RUN: not %clangxx -fsycl -foffload-lto=thin %s -### 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s
// CHECK-ERROR: unsupported option '-foffload-lto=thin' for target 'spir64-unknown-unknown'

// Verify we error when using the new offload driver but with device code split set to off.
// RUN: not %clangxx -fsycl --offload-new-driver -foffload-lto=thin -fsycl-device-code-split=off %s -### 2>&1 | FileCheck -check-prefix=CHECK-SPLIT-ERROR %s
// CHECK-SPLIT-ERROR: '-fsycl-device-code-split=off' is not supported when '-foffload-lto=thin' is set with '-fsycl'

// Verify there's no error and we see the expected cc1 flags and tool invocations with the new offload driver.
// RUN: %clangxx -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} -foffload-lto=thin %s -### 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SUPPORTED -implicit-check-not=-emit-only-kernels-as-entry-points %s
// CHECK-SUPPORTED: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" {{.*}} "-flto=thin" "-flto-unit"
// CHECK-SUPPORTED: sycl-post-link
// CHECK-SUPPORTED-NOT: -properties
// CHECK-SUPPORTED-NEXT: file-table-tform{{.*}}
// CHECK-SUPPORTED-NEXT: llvm-foreach{{.*}} "--" {{.*}}clang{{.*}} "-fsycl-is-device"{{.*}} "-flto=thin" "-flto-unit"
// CHECK-SUPPORTED-NEXT: file-table-tform{{.*}}
// CHECK-SUPPORTED-NEXT: llvm-offload-binary{{.*}} "-o" "{{.*}}" "--image=file=@{{.*}}"
// CHECK-SUPPORTED: clang-linker-wrapper{{.*}} "-sycl-thin-lto"

// Verify host-only mode syntax-only compiles the source, not the ThinLTO foreach.
// RUN: %clangxx -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} -foffload-lto=thin \
// RUN:   --offload-host-only --no-offloadlib -ccc-print-phases -c %s 2>&1 | \
// RUN: FileCheck -check-prefix=HOST-ONLY-PHASES %s
// HOST-ONLY-PHASES: [[#D:]]: input, "{{.+}}sycl-lto.cpp", c++, (device-sycl)
// HOST-ONLY-PHASES-NEXT: [[#D+1]]: compiler, {[[#D]]}, none, (device-sycl)
// HOST-ONLY-PHASES-NOT: foreach
// HOST-ONLY-PHASES-NOT: linker

// RUN: %clangxx -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} -foffload-lto=thin \
// RUN:   --offload-host-only --no-offloadlib -### -c %s 2>&1 | \
// RUN: FileCheck -check-prefix=HOST-ONLY-CMDS %s
// HOST-ONLY-CMDS: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" {{.*}} "-fsycl-is-device"
// HOST-ONLY-CMDS-SAME: "-fsyntax-only"
// HOST-ONLY-CMDS-NOT: sycl-post-link
// HOST-ONLY-CMDS-NOT: llvm-foreach
// HOST-ONLY-CMDS: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-host"
