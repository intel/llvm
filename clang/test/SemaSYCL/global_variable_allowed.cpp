// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -std=c++17 -sycl-std=2020 -verify -fsyntax-only %s

#include "sycl.hpp"

// This test shows that global_variable_allowed attribute allows
// global variables of type decorated with it to be referenced in device code.

template <typename T>
struct [[__sycl_detail__::global_variable_allowed]] global_variable_allowed {
public:
  const T &get() const noexcept { return *Data; }
  global_variable_allowed() {}
  operator T &() noexcept { return *Data; }

private:
  T *Data;
};

global_variable_allowed<int> glob;
static global_variable_allowed<float> static_glob;
inline global_variable_allowed<double> inline_glob;
static const global_variable_allowed<int> static_const_glob;

struct Foo {
  static global_variable_allowed<char> d;
};
global_variable_allowed<char> Foo::d;
global_variable_allowed<Foo> foo_instance;
static global_variable_allowed<Foo> foo_static_instance;

extern global_variable_allowed<int> ext;

template <typename T>
struct Bar {
  static global_variable_allowed<T> e;
};
static const Bar<int> f;

namespace baz {
  global_variable_allowed<int> baz_glob;
}

// expected-error@+1{{'global_variable_allowed' attribute only applies to classes}}
[[__sycl_detail__::global_variable_allowed]] int integer;

// expected-error@+1{{'global_variable_allowed' attribute only applies to classes}}
[[__sycl_detail__::global_variable_allowed]] int *pointer;

// expected-error@+1{{'global_variable_allowed' attribute takes no arguments}}
struct [[__sycl_detail__::global_variable_allowed(72)]] attribute_argument;


int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task<class KernelName1>([=]() {
      (void)glob;
      (void)static_glob;
      (void)inline_glob;
      (void)static_const_glob;
      (void)Foo::d;
      (void)foo_instance;
      (void)foo_static_instance;
      (void)ext;
      (void)f;
    });
  });
}
