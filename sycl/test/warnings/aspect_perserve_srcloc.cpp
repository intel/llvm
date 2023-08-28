// RUN: %clangxx -fsycl-device-only -fsycl-early-optimizations -Xclang -verify %s

// Tests for the preservation of source location information when warning
// diagnostics are issued due to mismatches in aspects used and those specified
// as available through device_has.

#include <sycl/sycl.hpp>

// expected-note@+1 {{propagated from call to function 'Struct1::Struct1()'}}
struct [[__sycl_detail__::__uses_aspects__(sycl::aspect::fp16)]] Struct1 {
  int a = 0;
};

// expected-note@+1 {{propagated from call to function 'func1(int)'}}
[[__sycl_detail__::__uses_aspects__(sycl::aspect::cpu)]] int func1(int a) {
  return 0;
}

// expected-note@+1 {{propagated from call to function 'func2(int)'}}
int func2(int a) {
  Struct1 s;
  s.a = 1;
  return s.a;
}

// expected-note@+1 2 {{propagated from call to function 'func3(int)'}}
int func3(int a) { return func1(a) + func2(a); }

int main() {
  sycl::queue Q;
  // expected-warning-re@+3 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'sycl::device_has' attribute}}
  // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'sycl::device_has' attribute}}
  Q.single_task(
      []() [[sycl::device_has(sycl::aspect::fp64)]] { (void)func3(1); });
  return 0;
}
