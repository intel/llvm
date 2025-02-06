// RUN: %clangxx -fsycl -c -Xclang -verify %s -o %t.ignored
//
// This test is intended to check that no diagnostics are emitted when a kernel
// performing virtual function calls is submitted with the right properties.
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

void foo(Base *Ptr) {
  Ptr->foo();
}

void bar(SubDerived *Ptr) {
  Ptr->bar();
}

// The exact arguments passed to calls_indirectly property don't matter,
// because we have no way of connecting a virtual call with a particular
// property set, but we test different properties here just in case.
oneapi::properties props_empty{oneapi::assume_indirect_calls};
oneapi::properties props_void{oneapi::assume_indirect_calls_to<void>};
oneapi::properties props_int{oneapi::assume_indirect_calls_to<int>};
oneapi::properties props_base{oneapi::assume_indirect_calls_to<Base>};

struct TestKernel_props_empty {
  char *Storage;
  TestKernel_props_empty(char *Storage_param) { Storage = Storage_param; };
  void operator()() const {
    new (Storage) SubSubDerived;
    auto *Ptr = reinterpret_cast<Base *>(Storage);

    Ptr->foo();
  }
  auto get(oneapi::properties_tag) { return props_empty; }
};

struct TestKernel_props_void {
  char *Storage;
  TestKernel_props_void(char *Storage_param) { Storage = Storage_param; };
  void operator()() const {
    new (Storage) SubDerived;
    auto *Ptr = reinterpret_cast<Derived *>(Storage);

    Ptr->bar();
  }
  auto get(oneapi::properties_tag) { return props_void; }
};

struct TestKernel_props_int {
  char *Storage;
  TestKernel_props_int(char *Storage_param) { Storage = Storage_param; };
  void operator()() const {
    new (Storage) Derived;
    auto *Ptr = reinterpret_cast<Base *>(Storage);
    foo(Ptr);
  }
  auto get(oneapi::properties_tag) { return props_int; }
};

struct TestKernel_props_base {
  char *Storage;
  TestKernel_props_base(char *Storage_param) { Storage = Storage_param; };
  void operator()() const {
    auto *Ptr = reinterpret_cast<SubDerived *>(Storage);
    bar(Ptr);
  }
  auto get(oneapi::properties_tag) { return props_base; }
};

int main() {
  sycl::queue q;

  char *Storage = sycl::malloc_device<char>(128, q);

  q.single_task(TestKernel_props_empty(Storage));
  q.single_task(TestKernel_props_void(Storage));
  q.single_task(TestKernel_props_int(Storage));
  q.single_task(TestKernel_props_base(Storage));

  sycl::free(Storage, q);

  return 0;
}
