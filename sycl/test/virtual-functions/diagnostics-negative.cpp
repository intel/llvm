// RUN: %clangxx -fsycl -fsycl-device-only -c -Xclang -verify %s -o %t.ignored
//
// This test is intended to check that correct diagnostics are emitted if
// kernel attempts to perform virtual calls, but wasn't submitted with the right
// properties.

#include <sycl/sycl.hpp>

namespace oneapi = sycl::ext::oneapi::experimental;

class Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void foo() const {}

  virtual void bar() const {}
};

class Derived : public Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void foo() const override {}

  void bar() const override {}
};

void foo(Base *Ptr) { // expected-note {{performed by function 'foo(Base*)'}}
  Ptr->foo();
}

void bar(Base *Ptr) { // expected-note {{performed by function 'bar(Base*)'}}
  Ptr->bar();
}

int main() {
  sycl::queue q;

  Base *Ptr = sycl::malloc_device<Derived>(1, q);

  // expected-note@+1 {{performed by function 'main::'lambda'()::operator()() const'}}
  q.single_task([=]() { // expected-error {{kernel 'typeinfo name for main::'lambda'()' performs a virtual function call}}
    Ptr->foo();
  });

  // expected-note@+1 {{performed by function 'main::'lambda0'()::operator()() const'}}
  q.single_task<class Kernel1>([=]() { // expected-error {{kernel 'typeinfo name for main::Kernel1' performs a virtual function call}}
    foo(Ptr);
  });

  // expected-note@+1 2{{performed by function 'main::'lambda1'()::operator()() const'}}
  q.single_task<class Kernel2>([=]() { // expected-error 2{{kernel 'typeinfo name for main::Kernel2' performs a virtual function call}}
    bar(Ptr);
    Ptr->bar();
  });

  sycl::free(Ptr, q);

  return 0;
}

