// REQUIRES: native_cpu_ock
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-dump-device-ir %s | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -O2 -mllvm -sycl-native-cpu-vecz-width=16 -mllvm -sycl-native-dump-device-ir %s | FileCheck %s --check-prefix=CHECK-16
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -O2 -mllvm -sycl-native-cpu-vecz-width=4 -mllvm -sycl-native-dump-device-ir %s | FileCheck %s --check-prefix=CHECK-4
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -O0 -mllvm -sycl-native-dump-device-ir %s | FileCheck %s --check-prefix=CHECK-O0
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -O2 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck %s --check-prefix=CHECK-DISABLE

// Invalid invocations: check that they don't crash the compiler
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-dump-device-ir %s > %t_temp.ll
// RUN: opt -O3 -verify-each %t_temp.ll -S -o %t_temp2.ll
// RUN: %clangxx -O2 -mllvm -sycl-native-cpu-backend -S -emit-llvm %t_temp.ll

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
