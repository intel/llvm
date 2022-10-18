/// Check if -fsycl-instrument-device-code is passed to device-side -cc1
/// and if ITT device libraries are pulled in.
/// The following conditions must be fulfilled:
/// 1. A SPIR-V-based environment must be targetted
/// 2. The option must not be explicitly disabled in the Driver call

/// FIXME: Force linux targets to allow for the libraries to be found.  Dummy
/// inputs for --sysroot should be updated to work better for Windows.
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --sysroot=%S/Inputs/SYCL -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-SPIRV,CHECK-HOST %s
/// -fno-sycl-device-lib mustn't affect the linkage of ITT libraries
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --sysroot=%S/Inputs/SYCL -fno-sycl-device-lib=all -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-SPIRV %s

// CHECK-SPIRV: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-instrument-device-code"
// CHECK-SPIRV: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.{{o|obj}}" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.{{o|obj}}" "-unbundle"
// CHECK-SPIRV: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.{{o|obj}}" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.{{o|obj}}" "-unbundle"
// CHECK-SPIRV: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.{{o|obj}}" "-output={{.*}}libsycl-itt-stubs-{{.*}}.{{o|obj}}" "-unbundle"
// CHECK-SPIRV: llvm-link{{.*}} "-only-needed" "{{.*}}" "-o" "{{.*}}.bc" "--suppress-warnings"
// CHECK-HOST-NOT: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-fsycl-instrument-device-code"

// RUN: %clangxx -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-NONPASSED %s
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fno-sycl-instrument-device-code -nocudalib -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-NONPASSED %s
// CHECK-NONPASSED-NOT: "-fsycl-instrument-device-code"
// CHECK-NONPASSED-NOT: "-input={{.*}}libsycl-itt-{{.*}}.{{o|obj}}"
