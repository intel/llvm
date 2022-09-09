/// Verify the usage of -fsycl-force-target applies to all expected unbundlings
// RUN: touch %t.o
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -fsycl-force-target=spir64 \
// RUN:          %s --sysroot=%S/Inputs/SYCL %t.o -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK_FORCE_TARGET
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen \
// RUN:          -fsycl-force-target=spir64-unknown-unknown \
// RUN:          %s --sysroot=%S/Inputs/SYCL %t.o -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=CHECK_FORCE_TARGET,CHECK_FORCE_TARGET_GEN
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 \
// RUN:          -fsycl-force-target=spir64 %s \
// RUN:          --sysroot=%S/Inputs/SYCL %t.o -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=CHECK_FORCE_TARGET,CHECK_FORCE_TARGET_CPU
// CHECK_FORCE_TARGET: clang-offload-bundler{{.*}} "-type=o" "-targets=host-{{.*}},sycl-spir64-unknown-unknown" "-input={{.*}}" "-output={{.*}}" "-output=[[DEVICEOBJECTOUT:.+]]" "-unbundle" "-allow-missing-bundles"
// CHECK_FORCE_TARGET: spirv-to-ir-wrapper{{.*}} "[[DEVICEOBJECTOUT]]" "-o" "[[DEVICEOBJECTBC:.+\.bc]]"
// CHECK_FORCE_TARGET: llvm-link{{.*}} "[[DEVICEOBJECTBC]]"{{.*}} "-o" "[[DEVICEOBJLINKED:.+\.bc]]" "--suppress-warnings"
// CHECK_FORCE_TARGET: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex{{.*}}" "-output={{.*}}libsycl-complex-{{.*}}" "-unbundle"
// CHECK_FORCE_TARGET_GEN: llvm-foreach{{.*}} {{.*}}ocloc{{.*}}
// CHECK_FORCE_TARGET_CPU: llvm-foreach{{.*}} {{.*}}opencl-aot{{.*}}

/// -fsycl-force-target is only valid with -fsycl-target with single targets
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen,spir64_x86_64 \
// RUN:          -fsycl-force-target=spir64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MULTIPLE_TARGET
// MULTIPLE_TARGET: error: multiple target usage with '-fsycl-targets=spir64_gen,spir64_x86_64' is not supported with '-fsycl-force-target=spir64'

