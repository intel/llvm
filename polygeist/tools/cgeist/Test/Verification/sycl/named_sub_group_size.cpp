// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

using namespace sycl;

class FunctorAuto {
public:
  [[intel::named_sub_group_size(automatic)]] void operator()() const {}
};

class FunctorPrim {
public:
  [[intel::named_sub_group_size(primary)]] void operator()() const {}
};

int main() {
  queue q;
  q.submit([&](handler &h) {
    FunctorAuto fauto;
    h.single_task<class kernel_name1>(fauto);

    FunctorPrim fprim;
    h.single_task<class kernel_name2>(fprim);
  });
  return 0;
}

// CHECK-MLIR: kernel_name1
// CHECK-MLIR-SAME:         intel_reqd_sub_group_size = "automatic"
// CHECK-MLIR: kernel_name2
// CHECK-MLIR-SAME:         intel_reqd_sub_group_size = "primary"

// CHECK-LLVM: define {{.*}}kernel_name1()
// CHECK-LLVM-SAME:                        !intel_reqd_sub_group_size ![[SGAUTO:[0-9]+]]
// CHECK-LLVM: define {{.*}}kernel_name2()
// CHECK-LLVM-SAME:                        !intel_reqd_sub_group_size ![[SGPRIM:[0-9]+]]
// CHECK-LLVM: ![[SGAUTO]] = !{!"automatic"}
// CHECK-LLVM: ![[SGPRIM]] = !{!"primary"}
