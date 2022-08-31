// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -verify %s
#include "Inputs/sycl.hpp"

// Diagnostic tests for device_global and global_variable_allowed attribute.

// Test that there are no errors when variables of type device_global are
// decorated with global_variable_allowed attribute appropriately.
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

protected:
  // expected-error@+1{{'device_global' member variable 'g' should be publicly accessible from namespace scope}}
  static device_global<int> g;
};

device_global<int> Baz::f;

device_global<int[4]> not_array; // OK

// expected-error@+1{{'device_global' array is not allowed}}
device_global<int> array[4];

device_global<int> same_name; // OK

namespace foo {
device_global<int> same_name; // OK
}

struct BBar {
private:
  struct BarInsider {
    // expected-error@+1{{'device_global' member variable 'c' should be publicly accessible from namespace scope}}
    static device_global<float> c;
  };

protected:
  struct BarInsiderProtected {
    // expected-error@+1{{'device_global' member variable 'c' should be publicly accessible from namespace scope}}
    static device_global<float> c;
  };
};

struct ABar {
  void method() {
    // expected-error@+1{{'device_global' variable must be a static data member or declared in global or namespace scope}}
    static device_global<float> c;
  }
  struct BarInsider {
    static device_global<float> c;
    void method() {
      // expected-error@+1{{'device_global' variable must be a static data member or declared in global or namespace scope}}
      static device_global<float> c;
    }
  };
};

template <typename T> void fooBar() {
  // expected-error@+1{{'device_global' variable must be a static data member or declared in global or namespace scope}}
  static device_global<T> c;
  // expected-error@+1{{'device_global' variable must be a static data member or declared in global or namespace scope}}
  device_global<T> d;
}

template <typename T> struct TS {
private:
  // expected-error@+1 2{{'device_global' member variable 'a' should be publicly accessible from namespace scope}}
  static device_global<T> a;
  // expected-error@+1 2{{'device_global' variable must be a static data member or declared in global or namespace scope}}
  device_global<T> b;
  // expected-error@+2{{'device_global' member variable 'c' should be publicly accessible from namespace scope}}
  // expected-error@+1 2{{'device_global' variable must be a static data member or declared in global or namespace scope}}
  device_global<int> c;

public:
  static device_global<T> d;
  // expected-error@+1 2{{'device_global' variable must be a static data member or declared in global or namespace scope}}
  device_global<T> e;
  // expected-error@+1 2{{'device_global' variable must be a static data member or declared in global or namespace scope}}
  device_global<int> f;

protected:
  // expected-error@+1 2{{'device_global' member variable 'g' should be publicly accessible from namespace scope}}
  static device_global<T> g;
  // expected-error@+1 2{{'device_global' variable must be a static data member or declared in global or namespace scope}}
  device_global<T> h;
  // expected-error@+2{{'device_global' member variable 'i' should be publicly accessible from namespace scope}}
  // expected-error@+1 2{{'device_global' variable must be a static data member or declared in global or namespace scope}}
  device_global<int> i;
};

// expected-note@+1{{in instantiation of template class 'TS<int>' requested here}}
TS<int> AAAA;

//expected-note@+2{{in instantiation of template class 'TS<char>' requested here}}
template <typename T> void templFoo () {
  TS<T> Var;
}

// expected-error@+2{{'device_global' attribute only applies to classes}}
// expected-error@+1{{'global_variable_allowed' attribute only applies to classes}}
[[__sycl_detail__::device_global]] [[__sycl_detail__::global_variable_allowed]] int integer;

// expected-error@+2{{'device_global' attribute only applies to classes}}
// expected-error@+1{{'global_variable_allowed' attribute only applies to classes}}
[[__sycl_detail__::device_global]] [[__sycl_detail__::global_variable_allowed]] int *pointer;

union [[__sycl_detail__::device_global]] [[__sycl_detail__::global_variable_allowed]] a_union;

int main() {
  // expected-note@+1{{in instantiation of function template specialization 'templFoo<char>' requested here}}
  templFoo<char>();

  // expected-note@+1{{in instantiation of function template specialization 'fooBar<int>' requested here}}
  fooBar<int>();

  sycl::kernel_single_task<class KernelName1>([=]() {
    (void)glob;
    (void)static_glob;
    (void)inline_glob;
    (void)static_const_glob;
    (void)Foo::d;
  });

  sycl::kernel_single_task<class KernelName2>([]() {
    // expected-error@+1{{'device_global' variable must be a static data member or declared in global or namespace scope}}
    device_global<int> non_static;
  });
}
