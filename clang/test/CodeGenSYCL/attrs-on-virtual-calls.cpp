// Test verifies that clang codegen properly adds call site attributes to
// device code

// RUN: %clang_cc1 -triple spir64 -fsycl-allow-virtual-functions \
// RUN:    -fsycl-is-device -emit-llvm %s -o %t.device
// RUN: FileCheck %s --input-file=%t.device
// RUN: %clang_cc1 -triple x86_64 -fsycl-allow-virtual-functions \
// RUN:    -fsycl-is-host -emit-llvm %s -o %t.host
// RUN: FileCheck %s --input-file=%t.host --check-prefix=CHECK-HOST

// CHECK-HOST-NOT: attributes {{.*}} "virtual-call"

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

SYCL_EXTERNAL bool rand();

class Base {
public:
  virtual void display() {}
};

class Derived : public Base {
public:
  void display() override {}

  // CHECK-LABEL: define {{.*}} @_ZN7Derived3foo
  // CHECK: call {{.*}}void %[[#]]{{.*}} #[[#VIRTUAL_CALL_ATTR:]]
  // CHECK: call {{.*}}void @_ZN4Base7display{{.*}} #[[#DIRECT_CALL_ATTR:]]
  void foo() {
    display(); // virtual call
    Base::display(); // non-virtual call
  }
};

SYCL_EXTERNAL void test_calls_with_implicit_this() {
  Derived d;
  d.foo();
}

// CHECK-LABEL: define {{.*}} @_Z21test_base_ref_to_base
// CHECK: call {{.*}}void %[[#]]{{.*}} #[[#VIRTUAL_CALL_ATTR]]
// CHECK: call {{.*}}void @_ZN4Base7display{{.*}} #[[#DIRECT_CALL_ATTR]]
SYCL_EXTERNAL void test_base_ref_to_base() {
  Base b;
  Base &br = b;

  br.display(); // virtual call
  br.Base::display(); // non-virtual call
}

// CHECK-LABEL: define {{.*}} @_Z24test_base_ref_to_derived
// CHECK: call {{.*}}void %[[#]]{{.*}} #[[#VIRTUAL_CALL_ATTR]]
// CHECK: call {{.*}}void @_ZN4Base7display{{.*}} #[[#DIRECT_CALL_ATTR]]
SYCL_EXTERNAL void test_base_ref_to_derived() {
  Derived d;
  Base &br = d;

  br.display(); // virtual call
  br.Base::display(); // non-virtual call
}

// CHECK-LABEL: define {{.*}} @_Z21test_base_ptr_to_base
// CHECK: call {{.*}}void %[[#]]{{.*}} #[[#VIRTUAL_CALL_ATTR]]
// CHECK: call {{.*}}void @_ZN4Base7display{{.*}} #[[#DIRECT_CALL_ATTR]]
SYCL_EXTERNAL void test_base_ptr_to_base() {
  Base b;
  Base *bp = &b;

  bp->display(); // virtual call
  bp->Base::display(); // non-virtual call
}

// CHECK-LABEL: define {{.*}} @_Z24test_base_ptr_to_derived
// CHECK: call {{.*}}void %[[#]]{{.*}} #[[#VIRTUAL_CALL_ATTR]]
// CHECK: call {{.*}}void @_ZN4Base7display{{.*}} #[[#DIRECT_CALL_ATTR]]
SYCL_EXTERNAL void test_base_ptr_to_derived() {
  Derived d;
  Base *bp = &d;

  bp->display(); // virtual call
  bp->Base::display(); // non-virtual call
}

// CHECK-LABEL: define {{.*}} @_Z9test_base
// CHECK: call {{.*}}void @_ZN4Base7display{{.*}} #[[#DIRECT_CALL_ATTR]]
// CHECK: call {{.*}}void @_ZN4Base7display{{.*}} #[[#DIRECT_CALL_ATTR]]
SYCL_EXTERNAL void test_base() {
  Base b;

  // Strictly speaking, this is a virtual function call, but clang emits it
  // as a direct call
  b.display();
  b.Base::display(); // non-virtual call
}

// CHECK-LABEL: define {{.*}} @_Z12test_derived
// CHECK: call {{.*}}void @_ZN7Derived7display{{.*}} #[[#DIRECT_CALL_ATTR]]
// CHECK: call {{.*}}void @_ZN4Base7display{{.*}} #[[#DIRECT_CALL_ATTR]]
SYCL_EXTERNAL void test_derived() {
  Derived d;

  // Strictly speaking, this is a virtual function call, but clang emits it
  // as a direct call
  d.display();
  d.Base::display(); // non-virtual call
}

// CHECK-NOT: attributes #[[#DIRECT_CALL_ATTR]] = {{.*}}"virtual-call"
// CHECK: attributes #[[#VIRTUAL_CALL_ATTR]] = {{.*}}"virtual-call"
