// Purpose of this test is to ensure that in SYCL device code mode, we are only
// emitting those virtual functions into vtable, which are marked with
// [[intel::device_indirectly_callable]] attribute or SYCL_EXTERNAL macro.
//
// RUN: %clang_cc1 -emit-llvm -o - -fsycl-is-device \
// RUN:     -fsycl-allow-virtual-functions -internal-isystem %S/Inputs \
// RUN:     -triple spir64 %s -o %t.ll
// RUN: FileCheck %s --input-file %t.ll --implicit-check-not host \
// RUN:     --implicit-check-not _ZN8Derived416maybe_device_barEv
//
// Check that vtable contains null pointers for host-only virtual functions:
//
// CHECK: @_ZTV4Base =
// CHECK-SAME:, {{.*}} @_ZN4Base10device_fooEv
// CHECK-SAME:, {{.*}} @_ZN4Base10device_barEv
// CHECK-SAME:, ptr addrspace(4) null
// CHECK-SAME:, ptr addrspace(4) null
//
// CHECK: @_ZTV8Derived1 =
// CHECK-SAME:, {{.*}} @_ZN8Derived110device_fooEv
// CHECK-SAME:, {{.*}} @_ZN4Base10device_barEv
// CHECK-SAME:, ptr addrspace(4) null
// CHECK-SAME:, ptr addrspace(4) null
// CHECK-SAME:, {{.*}} @_ZN8Derived110device_bazEv
// CHECK-SAME:, ptr addrspace(4) null
//
// CHECK: @_ZTV8Derived2 =
// CHECK-SAME:, {{.*}} @_ZN4Base10device_fooEv
// CHECK-SAME:, {{.*}} @_ZN8Derived210device_barEv
// CHECK-SAME:, ptr addrspace(4) null
// CHECK-SAME:, ptr addrspace(4) null
//
// CHECK: @_ZTV10SubDerived =
// CHECK-SAME:, {{.*}} @_ZN8Derived110device_fooEv
// CHECK-SAME:, {{.*}} @_ZN4Base10device_barEv
// CHECK-SAME:, ptr addrspace(4) null
// CHECK-SAME:, ptr addrspace(4) null
// CHECK-SAME:, {{.*}} @_ZN10SubDerived10device_bazEv
// CHECK-SAME:, ptr addrspace(4) null
//
// CHECK: @_ZTV8Derived3 =
// CHECK-SAME:, {{.*}} @_ZN8Derived310device_fooEv
// CHECK-SAME:, {{.*}} @_ZN8Derived316maybe_device_barEv
// CHECK-SAME:, ptr addrspace(4) null
//
// CHECK: @_ZTV12AbstractBase =
// CHECK-SAME: zeroinitializer
//
// CHECK: @_ZTV8Derived4 =
// CHECK-SAME:, {{.*}} @_ZN8Derived410device_fooEv
// CHECK-SAME:, ptr addrspace(4) null
// CHECK-SAME:, ptr addrspace(4) null
//
// Check that bodies of device virtual functions are present:
//
// CHECK-DAG: define linkonce_odr spir_func void @_ZN4Base10device_fooEv
// CHECK-DAG: define linkonce_odr spir_func void @_ZN4Base10device_barEv
//
// CHECK-DAG: define linkonce_odr spir_func void @_ZN8Derived110device_fooEv
// CHECK-DAG: define linkonce_odr spir_func void @_ZN8Derived110device_bazEv
//
// CHECK-DAG: define linkonce_odr spir_func void @_ZN10SubDerived10device_bazEv
//
// CHECK-DAG: define linkonce_odr spir_func void @_ZN8Derived210device_barEv
//
// CHECK-DAG: define linkonce_odr spir_func void @_ZN8Derived310device_fooEv
// CHECK-DAG: define linkonce_odr spir_func void @_ZN8Derived316maybe_device_barEv
//
// CHECK-DAG: define linkonce_odr spir_func void @_ZN8Derived410device_fooEv

#include "sycl.hpp"

struct Base {
  [[intel::device_indirectly_callable]] virtual void device_foo() {}
  [[intel::device_indirectly_callable]] virtual void device_bar() {}

  virtual void host_foo() {}
  virtual void host_bar() {}
};

struct Derived1 : public Base {
  [[intel::device_indirectly_callable]] void device_foo() override {}

  void host_bar() override {}

  [[intel::device_indirectly_callable]] virtual void device_baz() {}
  virtual void host_baz() {}
};

struct SubDerived : public Derived1 {
  [[intel::device_indirectly_callable]] void device_baz() override {}
  void host_baz() override {}
};

struct Derived2 : public Base {
  [[intel::device_indirectly_callable]] void device_bar() override {}

  void host_foo() override {}
};

class AbstractBase {
  virtual void device_foo() = 0;
  virtual void maybe_device_bar() = 0;
  virtual void host_foo() = 0;
};

class Derived3 : public AbstractBase {
  SYCL_EXTERNAL void device_foo() override {}
  SYCL_EXTERNAL void maybe_device_bar() override {}
  void host_foo() override {}
};

class Derived4 : public AbstractBase {
  SYCL_EXTERNAL void device_foo() override {}
  void host_foo() override {}
  void maybe_device_bar() override {}
};

int main(int argc, char *argv[]) {
  sycl::kernel_single_task<class kernel_function>([=]() {
    Base b;
    Derived1 d1;
    Derived2 d2;
    SubDerived sd;
    Base *ptr;
    if (argc > 5) {
      ptr = &d1;
    } else if (argc > 42) {
      ptr = &d2;
    } else {
      ptr = &sd;
    }

    Derived3 d3;
    Derived4 d4;
    AbstractBase *aptr;
    if (argc > 5) {
      aptr = &d3;
    } else {
      aptr = &d4;
    }
  });

  return 0;
}
