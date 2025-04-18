// REQUIRES: native_cpu
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -mllvm -sycl-native-dump-device-ir %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOAVX
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -march=skylake -fsycl -fsycl-targets=native_cpu -mllvm -sycl-native-dump-device-ir %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -mavx -fsycl -fsycl-targets=native_cpu -mllvm -sycl-native-dump-device-ir %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX

#include <sycl/sycl.hpp>
using namespace sycl;

class Test;
int main() {
  sycl::queue deviceQueue;
  deviceQueue.submit([&](handler &h) { h.single_task<Test>([=] {}); });
}

// CHECK: define void @_ZTS4Test.NativeCPUKernel({{.*}}) [[ATTRS:#[0-9]+]]
// CHECK: [[ATTRS]] = {
// NOAVX-NOT: "target-features"="{{[^"]*}}+avx{{[^"]*}}"
// AVX-SAME: "target-features"="{{[^"]*}}+avx{{[^"]*}}"
