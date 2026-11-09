// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that a local variable initialized within a
// parallel_for_work_group scope is initialized as an UndefValue in addrspace(3)
// in LLVM IR. It also checks that a private_memory variable stays a work-item
// scope alloca, and that a variable of a type with the wg_scope attribute is
// treated as a hierarchical work group local (no initializer).

#include "Inputs/sycl.hpp"

using namespace sycl;

namespace sycl {
template <typename T> struct __SYCL_TYPE(private_memory) private_memory {
  T Val;
};
} // namespace sycl

struct [[__sycl_detail__::wg_scope]] WG {
  int I;
};

int main() {
  queue q;
  q.submit([&](handler &h) {
    h.parallel_for_work_group<class kernel>(
        range<1>{1}, range<1>{1}, [=](group<1> G) {
          int WG_VAR = 10;
          private_memory<int> WI_VAR;
          WG WG_CLASS_VAR;
        });
  });
  return 0;
}

// CHECK-DAG: @{{.*}}WG_VAR = internal addrspace(3) global {{.*}} undef, {{.*}}
// CHECK-DAG: @{{.*}}WG_CLASS_VAR = internal addrspace(3) global %struct.WG undef, {{.*}}
// CHECK: %WI_VAR = alloca %"struct.sycl::private_memory"{{.*}}, !work_item_scope
