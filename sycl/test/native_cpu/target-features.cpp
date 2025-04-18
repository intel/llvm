// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOAVX
// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -aux-target-cpu skylake -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AVX
// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -aux-target-feature +avx -fsycl-is-device -fsycl-is-native-cpu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AVX

#if __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL void foo() {}
#endif

// CHECK: define void @_Z3foov() [[FOO_ATTRS:#[0-9]+]] {
// CHECK: [[FOO_ATTRS]] = {
// NOAVX-NOT: "target-features"="{{[^"]*}}+avx{{[^"]*}}"
// AVX-SAME: "target-features"="{{[^"]*}}+avx{{[^"]*}}"
