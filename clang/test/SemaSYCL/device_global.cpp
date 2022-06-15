// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -verify %s
#include "Inputs/sycl.hpp"

// Test cases below check for valid usage of device_global and
// global_variable_allowed attributes, and that they are being correctly
// generated in the AST.
using namespace sycl::ext::oneapi;

device_global<int> glob;                  // OK
static device_global<float> static_glob;  // OK
inline device_global<double> inline_glob; // OK
static const device_global<int> static_const_glob;

struct Foo {
  static device_global<char> d; // OK
};
device_global<char> Foo::d;

struct Baz {
private:
  // expected-error@+1{{'device_global' member variable 'f' should be publicly accessible from namespace scope}}
  static device_global<int> f;
};
device_global<int> Baz::f;

device_global<int[4]> not_array; // OK

device_global<int> same_name; // OK

namespace foo {
device_global<int> same_name; // OK
}

namespace {                   // #PREVIOUS
device_global<int> same_name; // OK
}

// TODO: add diagnostic for this case
inline namespace other {
device_global<int> same_name; // ILLEGAL: shadows "device_global" variable
} // namespace other

// expected-error@+2{{non-inline namespace cannot be reopened as inline}}
// expected-note@#PREVIOUS{{previous definition is here}}
inline namespace {
namespace foo {
}
} // namespace

struct BBar {
private:
  struct BarInsider {
    static device_global<float> c;
  };
};

struct ABar {
  void method() {
    // expected-error@+1{{'device_global' variables must be static or declared at namespace scope}}
    static device_global<float> c;
  }
  struct BarInsider {
    static device_global<float> c;
    void method() {
      // expected-error@+1{{'device_global' variables must be static or declared at namespace scope}}
      static device_global<float> c;
    }
  };
};

template <typename T> void fooBar() {
  static device_global<T> c;
  device_global<T> d;
}

template <typename T> struct TS {
private:
  static device_global<T> a;
  // expected-error@+1 {{'device_global' variables must be static or declared at namespace scope}}
  device_global<T> b;
  // FIXME: Why are both messages emitted
  // expected-error@+2 {{'device_global' member variable 'c' should be publicly accessible from namespace scope}}
  // expected-error@+1 {{'device_global' variables must be static or declared at namespace scope}}
  device_global<int> c;

public:
  static device_global<T> d;
  // expected-error@+1 {{'device_global' variables must be static or declared at namespace scope}}
  device_global<T> e;
  // expected-error@+1 {{'device_global' variables must be static or declared at namespace scope}}
  device_global<int> f;
};

// expected-note@+1 {{in instantiation of template class 'TS<int>' requested here}}
TS<int> AAAA;

// expected-error@+2{{'device_global' attribute only applies to classes}}
// expected-error@+1{{'global_variable_allowed' attribute only applies to classes}}
[[__sycl_detail__::device_global]] [[__sycl_detail__::global_variable_allowed]] int integer;

// expected-error@+2{{'device_global' attribute only applies to classes}}
// expected-error@+1{{'global_variable_allowed' attribute only applies to classes}}
[[__sycl_detail__::device_global]] [[__sycl_detail__::global_variable_allowed]] int *pointer;

union [[__sycl_detail__::device_global]] [[__sycl_detail__::global_variable_allowed]] a_union;

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
    // expected-error@+1{{'device_global' variables must be static or declared at namespace scope}}
    static device_global<int> non_const_static;
  });
}
