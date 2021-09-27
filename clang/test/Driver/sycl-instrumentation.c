/// Check if -fsycl-instrument-device-code is passed to device-side -cc1
/// and if ITT device libraries are pulled in.
/// The following conditions must be fulfilled:
/// 1. A SPIR-V-based environment must be targetted
/// 2. The corresponding Driver option must be enabled explicitly

// RUN: %clangxx -fsycl -fsycl-instrument-device-code -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-SPIRV,CHECK-HOST %s
/// -fno-sycl-device-lib mustn't affect the linkage of ITT libraries
// RUN: %clangxx -fsycl -fsycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-SPIRV %s

// CHECK-SPIRV: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-instrument-device-code"
// CHECK-SPIRV: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-inputs={{.*}}libsycl-itt-user-wrappers.{{o|obj}}" "-outputs={{.*}}libsycl-itt-user-wrappers-{{.*}}.{{o|obj}}" "-unbundle"
// CHECK-SPIRV: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-inputs={{.*}}libsycl-itt-compiler-wrappers.{{o|obj}}" "-outputs={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.{{o|obj}}" "-unbundle"
// CHECK-SPIRV: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-inputs={{.*}}libsycl-itt-stubs.{{o|obj}}" "-outputs={{.*}}libsycl-itt-stubs-{{.*}}.{{o|obj}}" "-unbundle"
// CHECK-SPIRV: llvm-link{{.*}} "-only-needed" "{{.*}}" "-o" "{{.*}}.bc" "--suppress-warnings"
// CHECK-HOST-NOT: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-fsycl-instrument-device-code"

// RUN: %clangxx -fsycl -fsycl-targets=spir64 -### %s 2>&1	\
// RUN: | FileCheck -check-prefixes=CHECK-NONPASSED %s
// RUN: %clangxx -fsycl -fsycl-targets=nvptx-nvidia-cuda -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-NONPASSED %s
// CHECK-NONPASSED-NOT: "-fsycl-instrument-device-code"
// CHECK-NONPASSED-NOT: "-inputs={{.*}}libsycl-itt-{{.*}}.{{o|obj}}"
