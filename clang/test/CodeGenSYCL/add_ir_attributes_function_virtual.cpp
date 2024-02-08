// RUN: %clang_cc1 -internal-isystem %S/Inputs -triple spir64-unknown-unknown -fsycl-is-device \
// RUN:    -fsycl-allow-virtual-functions -S -emit-llvm %s -o - | FileCheck %s

// Test IR generated for add_ir_attributes_function on virtual functions.

#include "sycl.hpp"

class Base {
public:
  virtual void testVirtual();
};

 [[__sycl_detail__::add_ir_attributes_function("PropBase", "PropVal")]]
void Base::testVirtual() {}

class Derived1: public Base {
public:
  [[__sycl_detail__::add_ir_attributes_function("PropDerived", "PropVal")]]
  void testVirtual() override{}
};

class Derived2: public Base {
public:
  virtual void testVirtual() final{}
};

void foo() {
  sycl::queue deviceQueue;
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel>([]() {
     Base b;
     b.testVirtual();
     Derived1 d1;
     d1.testVirtual();
     Derived2 d2;
     d2.testVirtual();
    });
  });
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel()
// CHECK: define {{.*}}spir_func void @_ZN4Base11testVirtualEv{{.*}} #[[BaseAttrs:[0-9]+]]
// CHECK: define {{.*}}spir_func void @_ZN8Derived111testVirtualEv{{.*}} #[[Derived1Attrs:[0-9]+]]
// CHECK: define {{.*}}spir_func void @_ZN8Derived211testVirtualEv{{.*}} #[[Derived2Attrs:[0-9]+]]
// CHECK: attributes #[[Derived2Attrs]] = {
// CHECK-NOT: PropBase
// CHECK-NOT: PropDerived
// CHECK: }
// CHECK: attributes #[[BaseAttrs]] = { {{.*}}"PropBase"="PropVal"{{.*}} }
// CHECK: attributes #[[Derived1Attrs]] = { {{.*}}"PropDerived"="PropVal"{{.*}} }