// RUN: %clangxx -fsycl -fsycl-device-only -emit-llvm -S -Xclang -fsycl-allow-virtual-functions %s -o %t.ll
// RUN: FileCheck %s < %t.ll
//
// This test is intended to check integration between SYCL headers and SYCL FE,
// i.e. to make sure that setting properties related to virtual functions will
// result in the right LLVM IR.
//
// This test is specifically focused on the indirectly_callable property.
//
// CHECK: define {{.*}} @_ZN4Base3fooEv{{.*}} #[[#ATTR_SET_DEFAULT:]]
// CHECK: define {{.*}} @_ZN7Derived3barEv{{.*}} #[[#ATTR_SET_DEFAULT]]
// CHECK: define {{.*}} @_ZN10SubDerived3barEv{{.*}} #[[#ATTR_SET_INT:]]
// CHECK: define {{.*}} @_ZN13SubSubDerived3foo{{.*}} #[[#ATTR_SET_DEFAULT]]
// CHECK: define {{.*}} @_ZN7Derived3fooEv{{.*}} #[[#ATTR_SET_DEFAULT]]
// CHECK: define {{.*}} @_ZN13SubSubDerived3barEv{{.*}} #[[#ATTR_SET_BASE:]]
//
// CHECK-DAG: attributes #[[#ATTR_SET_DEFAULT]] {{.*}} "indirectly-callable"="_ZTSv"
// CHECK-DAG: attributes #[[#ATTR_SET_INT]] {{.*}} "indirectly-callable"="_ZTSi"
// CHECK-DAG: attributes #[[#ATTR_SET_BASE]] {{.*}} "indirectly-callable"="_ZTS4Base"

#include <sycl/sycl.hpp>

namespace intel = sycl::ext::intel::experimental;
namespace oneapi = sycl::ext::oneapi::experimental;

class Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(intel::indirectly_callable<>)
  virtual int foo();
};

int Base::foo() { return 42; }

class Derived : public Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(intel::indirectly_callable<void>)
  virtual int bar();
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(intel::indirectly_callable<>)
  int foo() override;
};

int Derived::foo() { return 43; }

int Derived::bar() { return 0; }

class SubDerived : public Derived {
public:
  int foo() override { return 44; }

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(intel::indirectly_callable<int>)
  int bar() override;
};

int SubDerived::bar() { return 1; }

class SubSubDerived : public SubDerived {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(intel::indirectly_callable<>)
  int foo() override;

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(intel::indirectly_callable<Base>)
  int bar() override;
};

int SubSubDerived::foo() { return 45; }

int SubSubDerived::bar() { return 2; }

int main() {
  sycl::queue q;
  auto *obj = sycl::malloc_device<Base>(1, q);

  // Due to a bug in clang (that is being addressed in intel/llvm#13985) we have
  // to construct objects of the classes defined above, or otherwise they
  // methods won't be emitted into device code, even though the
  // indirectly_callable property requires that.
  // TODO: drop 'main' function completely from this test to simplify it
  oneapi::properties props_empty{intel::calls_indirectly<>};
  q.single_task(props_empty, [=]() { new (obj) Base; });
  q.single_task(props_empty, [=]() { new (obj) Derived; });
  q.single_task(props_empty, [=]() { new (obj) SubDerived; });
  q.single_task(props_empty, [=]() { new (obj) SubSubDerived; });

  return 0;
}
