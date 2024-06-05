// Test verifies that clang codegen properly adds call site attributes to
// device code

// RUN: %clang_cc1 -triple spir64 -fsycl-allow-virtual-functions \
// RUN:    -fsycl-is-device -emit-llvm %s -o %t.device
// RUN: FileCheck %s --input-file=%t.device --check-prefixes=CHECK,CHECK-DEVICE
// RUN: %clang_cc1 -triple x86_64 -fsycl-allow-virtual-functions \
// RUN:    -fsycl-is-host -emit-llvm %s -o %t.host
// RUN: FileCheck %s --input-file=%t.host --check-prefixes=CHECK,CHECK-HOST

// CHECK-LABEL: define {{.*}} @_Z4testv
// CHECK: call{{.*}}void %[[#]](ptr
// CHECK-DEVICE-SAME: #[[#ATTRS:]]
// CHECK-HOST-SAME: %[[#]]){{[[:space:]]}}
// CHECK-DEVICE: attributes #[[#ATTRS]] = {{.*}} "virtual-call"

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

SYCL_EXTERNAL bool rand();

class Base {
public:
  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "")]]
  virtual void display() {}
};

class Derived1 : public Base {
public:
  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "")]]
  void display() override {}
};

SYCL_EXTERNAL void test() {
  Derived1 d1;
  Base *b = nullptr;
  if (rand())
    b = &d1;
  b->display();
}

