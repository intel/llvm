// RUN: %clang_cc1 -fsycl-is-device -fsycl-allow-virtual-functions -internal-isystem %S/Inputs -verify -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify=novirtualfunction -fsyntax-only %s

// This test verifies that pure virtual functions are not
// diagnosed if undefined in base class.

// expected-no-diagnostics

#include "sycl.hpp"

sycl::queue deviceQueue;

class AbstractClass {
  public:
  virtual void testVF() = 0;
};

class Derived : public AbstractClass {
    void testVF() {}
};

void foo() {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class CallToUndefinedFnTester>([]() {
      Derived d;
      AbstractClass* obj = &d;
      obj->testVF(); // novirtualfunction-error {{SYCL kernel cannot call a virtual function}}
    });
  });
}
