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

void foo(Base *Ptr) {
  Ptr->foo();
}

void bar(SubDerived *Ptr) {
  Ptr->bar();
}

int main() {
  sycl::queue q;

  // The exact arguments passed to calls_indirectly property don't matter,
  // because we have no way of connecting a virtual call with a particular
  // property set, but we test different properties here just in case.
  oneapi::properties props_empty{oneapi::calls_indirectly<>};
  oneapi::properties props_void{oneapi::calls_indirectly<void>};
  oneapi::properties props_int{oneapi::calls_indirectly<int>};
  oneapi::properties props_base{oneapi::calls_indirectly<Base>};

  char *Storage = sycl::malloc_device<char>(128, q);

  q.single_task(props_empty, [=]() {
    new (Storage) SubSubDerived;
    auto *Ptr = reinterpret_cast<Base *>(Storage);

    Ptr->foo();
  });

  q.single_task(props_void, [=]() {
    new (Storage) SubDerived;
    auto *Ptr = reinterpret_cast<Derived *>(Storage);

    Ptr->bar();
  });

  q.single_task(props_int, [=]() {
    new (Storage) Derived;
    auto *Ptr = reinterpret_cast<Base *>(Storage);
    foo(Ptr);
  });

  q.single_task(props_base, [=]() {
    auto *Ptr = reinterpret_cast<SubDerived *>(Storage);
    bar(Ptr);
  });

  sycl::free(Storage, q);

  return 0;
}

