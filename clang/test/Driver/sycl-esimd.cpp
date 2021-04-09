/// Check that the warning is emmited for using "-fsycl-explicit-simd"
// RUN: %clang -### -fsycl-explicit-simd %s 2>&1 | FileCheck %s
// RUN: %clang -### -fno-sycl-explicit-simd %s 2>&1 | FileCheck %s
// CHECK: the flag '-f{{.*}}sycl-explicit-simd' has been deprecated and will be ignored
