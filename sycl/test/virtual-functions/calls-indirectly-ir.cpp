// RUN: %clangxx -fsycl -fsycl-device-only -emit-llvm -S %s -o %t.ll
// RUN: FileCheck %s < %t.ll
//
// This test is intended to check integration between SYCL headers and SYCL FE,
// i.e. to make sure that setting properties related to virtual functions will
// result in the right LLVM IR.
//
// This test is specifically focused on the calls_indirectly property.
//
// CHECK: define {{.*}}KEmpty{{.*}} #[[#ATTR_SET_DEFAULT:]]
// CHECK: define {{.*}}KInt{{.*}} #[[#ATTR_SET_INT:]]
// CHECK: define {{.*}}KVoid{{.*}} #[[#ATTR_SET_DEFAULT]]
// CHECK: define {{.*}}KUserDefined{{.*}} #[[#ATTR_SET_USER_DEFINED:]]
// TODO: update the check below
// As of now calls_indirectly_property takes into account only the first
// template argument ignoring the rest. This will be fixed in a follow-up
// patches and the test should be updated to reflect that, because current
// behavior is not correct.
// CHECK: define {{.*}}KMultiple{{.*}} #[[#ATTR_SET_INT]]
//
// CHECK-DAG: attributes #[[#ATTR_SET_DEFAULT]] {{.*}} "calls-indirectly"="_ZTSv"
// CHECK-DAG: attributes #[[#ATTR_SET_INT]] {{.*}} "calls-indirectly"="_ZTSi"
// CHECK-DAG: attributes #[[#ATTR_SET_USER_DEFINED]] {{.*}} "calls-indirectly"="_ZTS12user_defined"

#include <sycl/sycl.hpp>

namespace oneapi = sycl::ext::oneapi::experimental;

struct user_defined {
  int a;
  float b;
};

class KEmpty;
class KInt;
class KVoid;
class KUserDefined;
class KMultiple;

int main() {
  sycl::queue q;

  oneapi::properties props_empty{oneapi::calls_indirectly<>};
  oneapi::properties props_int{oneapi::calls_indirectly<int>};
  oneapi::properties props_void{oneapi::calls_indirectly<void>};
  oneapi::properties props_user_defined{oneapi::calls_indirectly<user_defined>};
  oneapi::properties props_multiple{
      oneapi::calls_indirectly<int, user_defined>};

  q.single_task<KEmpty>(props_empty, [=]() {});
  q.single_task<KInt>(props_int, [=]() {});
  q.single_task<KVoid>(props_void, [=]() {});
  q.single_task<KUserDefined>(props_user_defined, [=]() {});
  q.single_task<KMultiple>(props_multiple, [=]() {});

  return 0;
}
