// RUN: %clang -fsycl-device-only -S -Xclang -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
#include <algorithm>

void test() {
  int foo = 0x42;
  // CHECK: %[[FOO:[a-zA-Z0-9_]+]] = alloca i32, align 4
  int i = 43;
  // CHECK: %[[I:[a-zA-Z0-9_]+]] = alloca i32, align 4

  int *p1 = &foo;
  int *p2 = &i;
  // CHECK: %[[P1:[a-zA-Z0-9_]+]] = alloca i32 addrspace(4)*, align 8
  // CHECK: %[[P2:[a-zA-Z0-9_]+]] = alloca i32 addrspace(4)*, align 8
  // CHECK: %[[P1GEN:[a-zA-Z0-9_]+]] = addrspacecast i32 addrspace(4)** %[[P1]] to i32 addrspace(4)* addrspace(4)*
  // CHECK: %[[P2GEN:[a-zA-Z0-9_]+]] = addrspacecast i32 addrspace(4)** %[[P2]] to i32 addrspace(4)* addrspace(4)*

  std::swap(p1, p2);
  // CHECK: call spir_func void @_ZSt4swap{{.*}}(i32 addrspace(4)* addrspace(4)* align 8 dereferenceable(8) %[[P1GEN]], i32 addrspace(4)* addrspace(4)* align 8 dereferenceable(8) %[[P2GEN]])

  std::swap(foo, i);
  // CHECK: %[[FOOCAST:[a-zA-Z0-9_]+]] = addrspacecast i32* %[[FOO]] to i32 addrspace(4)*
  // CHECK: %[[ICAST:[a-zA-Z0-9_]+]] = addrspacecast i32* %[[I]] to i32 addrspace(4)*
  // CHECK: call spir_func void @_ZSt4swap{{.*}}(i32 addrspace(4)* align 4 dereferenceable(4) %[[FOOCAST]], i32 addrspace(4)* align 4 dereferenceable(4) %[[ICAST]])
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() { test(); });
  return 0;
}
