// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -ast-dump -verify %s | FileCheck %s
#include "Inputs/sycl.hpp"

using namespace sycl::ext::oneapi;

device_global<int> glob;           // OK
static device_global<float> static_glob;    // OK
inline device_global<double> inline_glob;    // OK
static const device_global<int> static_const_glob;

struct Foo {
  static device_global<char> d;  // OK
};
device_global<char> Foo::d;

struct Bar {
  device_global<int> e;         // ILLEGAL: non-static member variable not
};                              // allowed

struct Baz {
 private:
  // expected-error@+1{{'device_global' member variable 'f' is not publicly accessible from namespace scope}}
  static device_global<int> f;  // ILLEGAL: not publicly accessible from
};                              // namespace scope
device_global<int> Baz::f;

device_global<int[4]> not_array;        // OK
device_global<int> h[4];        // ILLEGAL: array of "device_global" not
                                // allowed

device_global<int> same_name;   // OK
namespace foo {
  device_global<int> same_name; // OK
}
namespace {
  device_global<int> same_name; // OK
}

int main() {
  cl::sycl::kernel_single_task<class KernelName1>([=]() {
    (void)glob;
    (void)static_glob;
    (void)inline_glob;
    (void)static_const_glob;
    (void)Foo::d;
  });

  cl::sycl::kernel_single_task<class KernelName2>([]() {
    // expected-error@+1{{'device_global' variables must be static or declared at namespace scope}}
    device_global<int> non_static;

    // expect no error on non_const_static declaration if decorated with
    // [[__sycl_detail__::global_variable_allowed]]
    static device_global<int> non_const_static;
  });
}
//
// CHECK: ClassTemplateDecl {{.*}} device_global
// CHECK: CXXRecordDecl {{.*}} struct device_global definition
// CHECK: SYCLDeviceGlobalAttr {{.*}}
// CHECK: SYCLGlobalVariableAllowedAttr {{.*}}
// CHECK: ClassTemplateSpecializationDecl {{.*}} struct device_global definition
// CHECK: SYCLDeviceGlobalAttr {{.*}}
// CHECK: SYCLGlobalVariableAllowedAttr {{.*}}

