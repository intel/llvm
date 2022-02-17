// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -ast-dump %s | FileCheck %s
#include "Inputs/sycl.hpp"

using namespace sycl::ext::oneapi;

device_global<int> a;           // OK
static device_global<float> b;    // OK
inline device_global<double> c;    // OK

struct Foo {
  static device_global<char> d;  // OK
};
device_global<char> Foo::d;

//struct Bar {
//  device_global<int> e;         // ILLEGAL: non-static member variable not
//};                              // allowed

//struct Baz {
// private:
//  static device_global<int> f;  // ILLEGAL: not publicly accessible from
//};                              // namespace scope
//device_global<int> Baz::f;

//device_global<int[4]> g;        // OK
//device_global<int> h[4];        // ILLEGAL: array of "device_global" not
                                // allowed

//device_global<int> same_name;   // OK
//namespace foo {
//  device_global<int> same_name; // OK
//}
//namespace {
//  device_global<int> same_name; // OK
//}

//inline namespace other {
//  device_global<int> same_name; // ILLEGAL: shadows "device_global" variable
//}                               // with same name in enclosing namespace scope

//inline namespace {
//  namespace foo {               // ILLEGAL: namespace name shadows "::foo"
//  }                             // namespace which contains "device_global"
                                // variable.
//}
//
// CHECK: ClassTemplateDecl {{.*}} device_global
// CHECK: CXXRecordDecl {{.*}} struct device_global definition
// CHECK: SYCLDetailDeviceGlobalAttr {{.*}}
// CHECK: SYCLDetailGlobalVariableAllowedAttr {{.*}}
// CHECK: ClassTemplateSpecializationDecl {{.*}} struct device_global definition
// CHECK: SYCLDetailDeviceGlobalAttr {{.*}}
// CHECK: SYCLDetailGlobalVariableAllowedAttr {{.*}}

// CHECK: VarDecl {{.*}} a 'device_global<int>':'sycl::ext::oneapi::device_global<int>' callinit
// CHECK: CXXConstructExpr {{.*}} 'device_global<int>':'sycl::ext::oneapi::device_global<int>' 'void ()'
// CHECK: VarDecl {{.*}} b 'device_global<float>':'sycl::ext::oneapi::device_global<float>' static callinit
// CHECK: CXXConstructExpr {{.*}} 'device_global<float>':'sycl::ext::oneapi::device_global<float>' 'void ()'
// CHECK: VarDecl {{.*}} c 'device_global<double>':'sycl::ext::oneapi::device_global<double>' inline callinit
// CHECK: CXXConstructExpr {{.*}} 'device_global<double>':'sycl::ext::oneapi::device_global<double>' 'void ()'
// CHECK: CXXRecordDecl {{.*}} struct Foo definition
// CHECK: VarDecl {{.*}} d 'device_global<char>':'sycl::ext::oneapi::device_global<char>' static
// CHECK: VarDecl {{.*}} d 'device_global<char>':'sycl::ext::oneapi::device_global<char>' callinit
// CHECK: CXXConstructExpr {{.*}} 'device_global<char>':'sycl::ext::oneapi::device_global<char>' 'void ()'

