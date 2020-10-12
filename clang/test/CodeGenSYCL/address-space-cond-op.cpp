// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// CHECK: [[STYPE:%.+]] = type { i16 }
struct S {
  unsigned short x;
};

S foo(bool cond, S &lhs, S rhs) {
// CHECK-LABEL:@_Z3foobR1SS_
// CHECK:   br i1 {{.+}}, label %[[BTRUE:.+]], label %[[BFALSE:.+]]
//
// CHECK: [[BTRUE]]:
// CHECK:   %[[LHS:.+]] = load [[STYPE]] addrspace(4)*, [[STYPE]] addrspace(4)**
// CHECK:   br label %[[BEND:.+]]
//
// CHECK: [[BFALSE]]:
// CHECK:   %[[RHS:.+]] = addrspacecast [[STYPE]]* {{.+}} to [[STYPE]] addrspace(4)*
// CHECK:   br label %[[BEND]]
//
// CHECK: [[BEND]]:
// CHECK:   %{{.+}} = phi [[STYPE]] addrspace(4)* [ %[[LHS]], %[[BTRUE]] ], [ %[[RHS]], %[[BFALSE]] ]
  S val = cond ? lhs : rhs;
  return val;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel<class fake_kernel>([]() {
    S lhs, rhs;
    foo(true, lhs, rhs);
  });
  return 0;
}
