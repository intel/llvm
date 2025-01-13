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
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual void foo() {}
};

class Derived : public Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void foo() override {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable_in<void>)
  virtual void bar() {}
};

class SubDerived : public Derived {
public:
  void foo() override {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable_in<int>)
  void bar() override {}
};

class SubSubDerived : public SubDerived {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void foo() override {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable_in<Base>)
  void bar() override {}
};

oneapi::properties props_empty{oneapi::assume_indirect_calls};
oneapi::properties props_void{oneapi::assume_indirect_calls_to<void>};
oneapi::properties props_int{oneapi::assume_indirect_calls_to<int>};
oneapi::properties props_base{oneapi::assume_indirect_calls_to<Base>};
oneapi::properties props_multiple{oneapi::assume_indirect_calls_to<int, Base>};

struct TestKernel_props_empty {
  void operator()() const {}
  auto get(oneapi::properties_tag) { return props_empty; }
};

struct TestKernel_props_void {
  void operator()() const {}
  auto get(oneapi::properties_tag) { return props_void; }
};

struct TestKernel_props_int {
  void operator()() const {}
  auto get(oneapi::properties_tag) { return props_int; }
};

struct TestKernel_props_base {
  void operator()() const {}
  auto get(oneapi::properties_tag) { return props_base; }
};

struct TestKernel_props_multiple {
  void operator()() const {}
  auto get(oneapi::properties_tag) { return props_multiple; }
};

int main() {
  sycl::queue q;

  oneapi::properties props_empty{oneapi::assume_indirect_calls};
  oneapi::properties props_void{oneapi::assume_indirect_calls_to<void>};
  oneapi::properties props_int{oneapi::assume_indirect_calls_to<int>};
  oneapi::properties props_base{oneapi::assume_indirect_calls_to<Base>};
  oneapi::properties props_multiple{
     oneapi::assume_indirect_calls_to<int, Base>};

  q.single_task(TestKernel_props_empty{});
  q.single_task(TestKernel_props_void{});
  q.single_task(TestKernel_props_int{});
  q.single_task(TestKernel_props_base{});
  q.single_task(TestKernel_props_multiple{});

  return 0;
}
