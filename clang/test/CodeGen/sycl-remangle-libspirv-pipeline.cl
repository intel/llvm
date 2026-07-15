// Test that -fsycl-remangle-libspirv flag enables SYCLRemangleLibspirvPass.

// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN: %s 2>&1 | FileCheck --check-prefixes=DEFAULT %s

// DEFAULT-NOT: sycl-remangle-libspirv

// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN:   -fsycl-remangle-libspirv \
// RUN: %s 2>&1 | FileCheck --check-prefixes=ENABLED %s

// ENABLED: sycl-remangle-libspirv

void test() {}
