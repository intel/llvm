// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes %s -emit-llvm -o -  | FileCheck %s
// Tests that static data members that are not constant-initialized are not
// emitted in device code.

// CHECK:%struct._ZTS1B.B = type { i8 }
// CHECK:%struct._ZTS1C.C = type { i32, i32 }
// CHECK:%struct._ZTS1D.D = type { i8 }
// CHECK:$_ZN2TTIiE2c1E = comdat any
// CHECK:$_ZN2TTIiE2d1E = comdat any
// CHECK:$_ZN2TTIiE4var1E = comdat any
// CHECK:$_ZN2TTIiE4var3E = comdat any
// CHECK:@_ZN1S2b1E = addrspace(1) global %struct._ZTS1B.B zeroinitializer, align 1
// CHECK:@_ZN1S2c1E = addrspace(1) constant %struct._ZTS1C.C { i32 2, i32 5 }, align 4
// CHECK:@_ZN1S2d1E = addrspace(1) constant %struct._ZTS1D.D zeroinitializer, align 1
// CHECK:@_ZN1S4var1E = addrspace(1) constant i32 1, align 4
// CHECK:@_ZN1S4var2E = available_externally addrspace(1) constant i32 3, align 4
// CHECK:@_ZN1S4var3E = addrspace(1) constant i32 4, align 4
// CHECK:@_ZN2TTIiE2b1E = external addrspace(1) global %struct._ZTS1B.B, align 1
// CHECK:@_ZN2TTIiE2c1E = linkonce_odr addrspace(1) constant %struct._ZTS1C.C { i32 2, i32 5 }, comdat, align 4
// CHECK:@_ZN2TTIiE2d1E = linkonce_odr addrspace(1) constant %struct._ZTS1D.D zeroinitializer, comdat, align 1
// CHECK:@_ZN2TTIiE4var1E = linkonce_odr addrspace(1) constant i32 1, comdat, align 4
// CHECK:@_ZN2TTIiE4var2E = available_externally addrspace(1) constant i32 3, align 4
// CHECK:@_ZN2TTIiE4var3E = linkonce_odr addrspace(1) constant i32 4, comdat, align 4
// CHECK:@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535
// CHECK-NOT: addrspacecast
// CHECK: define spir_kernel void @_ZTSZ4mainE11fake_kernel()

struct TestBaseType {};
struct B {
  B();
};
struct C {
  int i, j;
};
struct D {
};

struct S {
  static const B b1;
  static const int var1;
  static constexpr const int var2 = 3;
  static const int var3;
  static const C c1;
  static const D d1;
};
const B S::b1;
const C S::c1{2, 5};
const int S::var1 = 1;
const int S::var3 = 1 + 3;
const D S::d1;

template <typename T>
struct TT {
  static const B b1;
  static const int var1;
  static constexpr const int var2 = 3;
  static const int var3;
  static const C c1;
  static const D d1;
};
template <typename T>
const B TT<T>::b1;
template <typename T>
const C TT<T>::c1{2, 5};
template <typename T>
const int TT<T>::var1 = 1;
template <typename T>
const int TT<T>::var3 = 1 + 3;
template <typename T>
const D TT<T>::d1;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
  (void)S::b1;
  (void)S::c1;
  (void)S::d1;
  (void)S::var1;
  (void)S::var2;
  (void)S::var3;
  (void)TT<int>::b1;
  (void)TT<int>::c1;
  (void)TT<int>::d1;
  (void)TT<int>::var1;
  (void)TT<int>::var2;
  (void)TT<int>::var3;
}
int main() {
  kernel_single_task<class fake_kernel>([=]() {});
  return 0;
}
