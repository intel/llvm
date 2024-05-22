// REQUIRES: native_cpu_ock
// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -S -emit-llvm  -o %t_temp.ll %s
// RUN: %clangxx -O2 -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clangxx -O2 -mllvm -sycl-native-cpu-backend -mllvm -sycl-native-cpu-vecz-width=16 -S -emit-llvm -o - %t_temp.ll | FileCheck %s --check-prefix=CHECK-16
// RUN: %clangxx -O2 -mllvm -sycl-native-cpu-backend -mllvm -sycl-native-cpu-vecz-width=4 -S -emit-llvm -o - %t_temp.ll | FileCheck %s --check-prefix=CHECK-4
// RUN: %clangxx -O0 -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s --check-prefix=CHECK-O0
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -O2 -mllvm -sycl-native-cpu-backend -mllvm -sycl-native-cpu-no-vecz -S -emit-llvm -o - %t_temp.ll | FileCheck %s --check-prefix=CHECK-DISABLE
#include <sycl/sycl.hpp>
class Test1;
int main() {
  sycl::queue deviceQueue;
  sycl::accessor<int, 1, sycl::access::mode::write> acc;
  sycl::range<1> r(1);
  deviceQueue.submit([&](sycl::handler &h) {
    h.parallel_for<Test1>(r, [=](sycl::id<1> id) { acc[id[0]] = 42; });
    // CHECK-DEFAULT: store <8 x i32> <i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42>
    // CHECK-16: store <16 x i32> <i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42, i32 42>
    // CHECK-4: store <4 x i32> <i32 42, i32 42, i32 42, i32 42>
    // CHECK-O0: store i32 42
    // CHECK-O0-NOT: store <{{.*}}>
    // CHECK-DISABLE: store i32 42
    // CHECK-DISABLE-NOT: store <{{.*}}>
  });
}
