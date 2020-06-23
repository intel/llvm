// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes %s -emit-llvm -o -  | FileCheck %s
// Test that static initializers do not force the emission of globals on sycl device

// CHECK: %struct._ZTS16RegisterBaseInit.RegisterBaseInit = type { i8 }
// CHECK-NOT: $_ZN8BaseInitI12TestBaseTypeE15s_regbase_ncsdmE = comdat any
// CHECK: $_ZN8BaseInitI12TestBaseTypeE3varE = comdat any
// CHECK: @_ZN8BaseInitI12TestBaseTypeE9s_regbaseE = {{.*}} global %struct._ZTS16RegisterBaseInit.RegisterBaseInit
// CHECK-NOT: @_ZN8BaseInitI12TestBaseTypeE15s_regbase_ncsdmE = weak_odr addrspace(1) global %struct._ZTS16RegisterBaseInit.RegisterBaseInit zeroinitializer, comdat, align 1
// CHECK: @_ZN8BaseInitI12TestBaseTypeE3varE = weak_odr addrspace(1) constant i32 9, comdat, align 4
// CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @__cxx_global_var_init, i8* addrspacecast (i8 addrspace(1)* getelementptr inbounds (%struct._ZTS16RegisterBaseInit.RegisterBaseInit, %struct._ZTS16RegisterBaseInit.RegisterBaseInit addrspace(1)* @_ZN8BaseInitI12TestBaseTypeE9s_regbaseE, i32 0, i32 0) to i8*) }]
// CHECK-NOT: @_ZGVN8BaseInitI12TestBaseTypeE15s_regbase_ncsdmE = weak_odr global i64 0, comdat($_ZN8BaseInitI12TestBaseTypeE9s_regbaseE), align 8
// CHECK: define spir_kernel void @_ZTSZ4mainE11fake_kernel()
// CHECK: call spir_func void @"_ZZ4mainENK3$_0clE16RegisterBaseInit
// CHECK: declare spir_func void @_ZN16RegisterBaseInit3fooEv

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
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc(BaseInit<TestBaseType>::s_regbase);
}
int main() {
  kernel_single_task<class fake_kernel>([=](RegisterBaseInit s) {
    s.foo();
  });
  return 0;
}
