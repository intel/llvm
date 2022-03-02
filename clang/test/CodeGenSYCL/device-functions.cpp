// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

template <typename T>
T bar(T arg);

void foo() {
  int a = 1 + 1 + bar(1);
}

template <typename T>
T bar(T arg) {
  return arg;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() { foo(); });
  return 0;
}
// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE11fake_kernel()
// CHECK: define internal spir_func void @_ZZ4mainENKUlvE_clEv(%class.anon addrspace(4)* {{[^,]*}} %this)
// CHECK: define {{.*}}spir_func void @_Z3foov()
// CHECK: define linkonce_odr spir_func noundef i32 @_Z3barIiET_S0_(i32 noundef %arg)
