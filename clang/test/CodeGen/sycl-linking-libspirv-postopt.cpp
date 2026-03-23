// Test that SYCL correctly uses -mlink-builtin-bitcode-postopt to enable
// SYCLBuiltinRemanglePass early and LinkInModulesPass late (after optimization)

// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN: %s 2>&1 | FileCheck --check-prefixes=DEFAULT %s

// DEFAULT-NOT: sycl-builtin-remangle
// DEFAULT-NOT: LinkInModulesPass

// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN:   -mlink-builtin-bitcode-postopt \
// RUN:   -fsycl-is-device \
// RUN: %s 2>&1 | FileCheck --check-prefixes=POSTOPT %s

// POSTOPT: sycl-builtin-remangle
// POSTOPT: LinkInModulesPass

// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN:   -mno-link-builtin-bitcode-postopt \
// RUN: %s 2>&1 | FileCheck --check-prefixes=NO-POSTOPT %s

// NO-POSTOPT-NOT: LinkInModulesPass

// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm-bc -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes \
// RUN:   -mlink-builtin-bitcode-postopt \
// RUN:   -mno-link-builtin-bitcode-postopt \
// RUN: %s 2>&1 | FileCheck --check-prefixes=POSTOPT-THEN-NO %s

// POSTOPT-THEN-NO-NOT: LinkInModulesPass

void test() {}
