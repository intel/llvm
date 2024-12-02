// There ios no guarantee that special symbol @__cxa_pure_virtual is suppored
// by SYCL backend compiler, so we need to make sure that we don't emit it
// during SYCL device compilation.
//
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -emit-llvm %s -o - | FileCheck %s
//
// CHECK-NOT: @__cxa_pure_virtual

SYCL_EXTERNAL bool rand();

class Base {
public:
  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "a")]]
  virtual void display() {}

  virtual void pure_host() = 0;

  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "a")]]
  virtual void pure_device() = 0;
};

class Derived1 : public Base {
public:
  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "a")]]
  void display() override {}

  void pure_host() override {}

  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "a")]]
  void pure_device() override {}
};

SYCL_EXTERNAL void test() {
  Derived1 d1;
  Base *b = nullptr;
  if (rand())
    b = &d1;
  b->display();
}

