// RUN: %clang_cc1 -fsycl-is-device -fsycl-allow-virtual-functions -internal-isystem %S/Inputs -verify -fsyntax-only %s

// This test verifies that pure virtual functions are not
// diagnosed if undefined in base class.

#include "sycl.hpp"

sycl::queue deviceQueue;

class AbstractClass {
  public:
  virtual void testVF() = 0; //expected-note{{unimplemented pure virtual method 'testVF' in 'Derived2'}}
};

class Derived1 : public AbstractClass {
    void testVF() {}
};

class Derived2 : public AbstractClass {
};

class Derived3 : public AbstractClass {
    void testVF();
};

void foo() {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class CallToUndefinedFnTester>([]() {
      Derived1 d1;
      AbstractClass* obj1 = &d1;
      obj1->testVF(); // No diagnostic

      Derived2 d2; // expected-error{{variable type 'Derived2' is an abstract class}}

      Derived3 d3;
      AbstractClass* obj2 = &d3;
      // Ideally we would want to diagnose the missing SYCL_EXTERNAL macro here since
      // there is no definition for this function in this TU. However it is expensive
      // to do so in the frontend, and a diagnostic is not currently emitted.
      obj2->testVF();
    });
  });
}
