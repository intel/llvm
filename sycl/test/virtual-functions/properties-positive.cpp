// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
//
// This test is intended to check that we can successfully compile code that
// uses new properties from the virtual functions extension.
//
// expected-no-diagnostics

#include <sycl/sycl.hpp>

namespace oneapi = sycl::ext::oneapi::experimental;

class Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void foo() {}
};

class Derived : public Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void foo() override {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<void>)
  virtual void bar() {}
};

class SubDerived : public Derived {
public:
  void foo() override {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<int>)
  void bar() override {}
};

class SubSubDerived : public SubDerived {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void foo() override {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<Base>)
  void bar() override {}
};

int main() {
  sycl::queue q;

  static_assert(
      oneapi::is_property_key<oneapi::indirectly_callable_key>::value);
  static_assert(oneapi::is_property_key<oneapi::calls_indirectly_key>::value);

  oneapi::properties props_empty{oneapi::calls_indirectly<>};
  oneapi::properties props_void{oneapi::calls_indirectly<void>};
  oneapi::properties props_int{oneapi::calls_indirectly<int>};
  oneapi::properties props_base{oneapi::calls_indirectly<Base>};
  oneapi::properties props_multiple{oneapi::calls_indirectly<int, Base>};

  q.single_task(props_empty, [=]() {});
  q.single_task(props_void, [=]() {});
  q.single_task(props_int, [=]() {});
  q.single_task(props_base, [=]() {});
  q.single_task(props_multiple, [=]() {});

  return 0;
}
