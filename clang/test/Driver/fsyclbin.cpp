/// Tests behaviors of -fsyclbin

/// -fsyclbin is only used with the new offloading model.
// RUN: %clangxx -fsycl -fsyclbin --no-offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=UNUSED
// UNUSED: warning: argument unused during compilation: '-fsyclbin'

/// Check tool invocation contents.
// RUN: %clangxx -fsycl -fsyclbin --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS
// RUN: %clang_cl -fsycl -fsyclbin --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS
// CHECK_TOOLS: clang-offload-packager
// CHECK_TOOLS-SAME: --image=file={{.*}}.bc,triple=spir64-unknown-unknown
// CHECK_TOOLS-SAME: kind=sycl
// CHECK_TOOLS: clang-linker-wrapper
// CHECK_TOOLS-SAME: --syclbin

/// Check compilation phases, only device compile should be performed
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fsyclbin \
// RUN:   --offload-new-driver %s -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASES
// CHECK_PHASES: 0: input, "{{.*}}", c++, (device-sycl)
// CHECK_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHECK_PHASES: 2: compiler, {1}, ir, (device-sycl)
// CHECK_PHASES: 3: backend, {2}, ir, (device-sycl)
// CHECK_PHASES: 4: offload, "device-sycl (spir64-unknown-unknown)" {3}, ir
// CHECK_PHASES: 5: clang-offload-packager, {4}, image, (device-sycl)
// CHECK_PHASES: 6: clang-linker-wrapper, {5}, image, (device-sycl)
// CHECK_PHASES: 7: offload, "device-sycl (x86_64-unknown-linux-gnu)" {6}, none
