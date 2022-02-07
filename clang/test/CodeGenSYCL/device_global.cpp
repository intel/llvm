// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -std=c++17 -emit-llvm %s -o - | FileCheck %s
#include "Inputs/sycl.hpp"

using namespace sycl::ext::oneapi;
static device_global<int> Foo;

device_global<int> a;           // OK
static device_global<int> b;    // OK
//inline device_global<int> c;    // OK

struct Foo {
  static device_global<int> d;  // OK
};
device_global<int> Foo::d;

struct Bar {
  device_global<int> e;         // ILLEGAL: non-static member variable not
};                              // allowed

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
