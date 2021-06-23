// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes %s -emit-llvm -o -  | FileCheck %s
// Test that static initializers do not force the emission of globals on sycl device

// CHECK-NOT: $_ZN8BaseInitI12TestBaseTypeE15s_regbase_ncsdmE = comdat any
// CHECK: $_ZN8BaseInitI12TestBaseTypeE3varE = comdat any
// CHECK: @_ZN8BaseInitI12TestBaseTypeE3varE = weak_odr addrspace(1) constant i32 9, comdat, align 4
// CHECK-NOT: @_ZN8BaseInitI12TestBaseTypeE15s_regbase_ncsdmE = weak_odr addrspace(1) global %struct._ZTS16RegisterBaseInit.RegisterBaseInit zeroinitializer, comdat, align 1
// CHECK-NOT: @_ZGVN8BaseInitI12TestBaseTypeE15s_regbase_ncsdmE = weak_odr global i64 0, comdat($_ZN8BaseInitI12TestBaseTypeE9s_regbaseE), align 8
// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE11fake_kernel()
// CHECK: call spir_func void @_ZZ4mainENKUlvE_clEv

struct TestBaseType {};
struct RegisterBaseInit {
  __attribute__((sycl_device)) void foo();
  RegisterBaseInit();
};
template <class T>
struct BaseInit {
  static const RegisterBaseInit s_regbase;
  static RegisterBaseInit s_regbase_ncsdm;
  static const int var;
};
template <class T>
const RegisterBaseInit BaseInit<T>::s_regbase;
template <class T>
RegisterBaseInit BaseInit<T>::s_regbase_ncsdm;
template <class T>
const int BaseInit<T>::var = 9;
template struct BaseInit<TestBaseType>;
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}
int main() {
  kernel_single_task<class fake_kernel>([=]() {
  });
  return 0;
}
