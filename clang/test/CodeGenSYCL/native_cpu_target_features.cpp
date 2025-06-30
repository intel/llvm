// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -emit-llvm -fsycl-is-native-cpu -o - %s | FileCheck %s --check-prefixes=CHECK,NOAVX
// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -aux-target-cpu skylake -fsycl-is-device -emit-llvm -fsycl-is-native-cpu -o - %s | FileCheck %s --check-prefixes=CHECK,AVX
// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -aux-target-feature +avx -fsycl-is-device -emit-llvm -fsycl-is-native-cpu -o - %s | FileCheck %s --check-prefixes=CHECK,AVX
//
// This is not sensible but check that we do not crash.
// RUN: %clang_cc1 -triple native_cpu -aux-triple native_cpu -fsycl-is-device -emit-llvm -fsycl-is-native-cpu -o - %s | FileCheck %s --check-prefixes=CHECK,NOAVX

#include "Inputs/sycl.hpp"
using namespace sycl;

class Test;
int main() {
  sycl::queue deviceQueue;
  deviceQueue.submit([&](handler &h) { h.single_task<Test>([=] {}); });
}

// CHECK: void @_ZTS4Test() [[ATTRS:#[0-9]+]]
// CHECK: [[ATTRS]] = {
// NOAVX-NOT: "target-features"="{{[^"]*}}+avx{{[^"]*}}"
// AVX-SAME: "target-features"="{{[^"]*}}+avx{{[^"]*}}"
