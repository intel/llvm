// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

using namespace sycl;

class Functor16 {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

template <int SIZE>
class Functor2 {
public:
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() const {}
};

int main() {
  queue q;
  q.submit([&](handler &h) {
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    h.single_task<class kernel_name2>(
        []() [[intel::reqd_sub_group_size(4)]]{});

    Functor2<2> f2;
    h.single_task<class kernel_name3>(f2);
  });
  return 0;
}

// CHECK-MLIR: kernel_name1
// CHECK-MLIR-SAME:         intel_reqd_sub_group_size = 16
// CHECK-MLIR: kernel_name2
// CHECK-MLIR-SAME:         intel_reqd_sub_group_size = 4
// CHECK-MLIR: kernel_name3
// CHECK-MLIR-SAME:         intel_reqd_sub_group_size = 2

// CHECK-LLVM: define {{.*}}kernel_name1()
// CHECK-LLVM-SAME:                        !intel_reqd_sub_group_size ![[SGSIZE16:[0-9]+]]
// CHECK-LLVM: define {{.*}}kernel_name2()
// CHECK-LLVM-SAME:                        !intel_reqd_sub_group_size ![[SGSIZE4:[0-9]+]]
// CHECK-LLVM: define {{.*}}kernel_name3()
// CHECK-LLVM-SAME:                        !intel_reqd_sub_group_size ![[SGSIZE2:[0-9]+]]
// CHECK-LLVM: ![[SGSIZE16]] = !{i32 16}
// CHECK-LLVM: ![[SGSIZE4]] = !{i32 4}
// CHECK-LLVM: ![[SGSIZE2]] = !{i32 2}
