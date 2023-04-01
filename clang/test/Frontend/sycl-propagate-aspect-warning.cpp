// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -mconstructor-aliases -no-opaque-pointers -emit-codegen-only -verify %s

// Tests for warnings when propagated aspects do not match the aspects available
// in a function, as specified through the 'sycl::device_has' attribute.

#include "sycl.hpp"

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}Struct1::Struct1({{.*}})'}}
struct [[__sycl_detail__::__uses_aspects__(sycl::aspect::fp16)]] Struct1 {
  int a = 0;
};

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func5(int)'}}
[[__sycl_detail__::__uses_aspects__(sycl::aspect::cpu)]] int func5(int a) {
  return 0;
}

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func3(int, int, int)'}}
int func3(int a, int b, int c) {
  Struct1 s;
  s.a = 1;
  return s.a;
}

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func4(int, int)'}}
int func4(int a, int b) {
  double x = 3.0;
  return func3(a, b, (int)x);
}

// expected-note-re@+1 2 {{propagated from call to function '{{.*}}func2(int, int)'}}
int func2(int a, int b) { return func3(a, b, 1); }

// expected-warning-re@+1 {{function '{{.*}}func1(int)' uses aspect 'fp16' not listed in its 'sycl::device_has' attribute}}
[[sycl::device_has(sycl::aspect::fp64)]] int func1(int a) { return func2(a, 1); }

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func6(int, int, int)'}}
int func6(int a, int b, int c) {
  return func5(a);
}

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func7(int, int)'}}
int func7(int a, int b) {
  double x = 3.0;
  return func6(a, b, (int)x);
}

// expected-note-re@+1 2 {{propagated from call to function '{{.*}}func8(int, int)'}}
int func8(int a, int b) { return func6(a, b, 1); }

// expected-warning-re@+1 {{function '{{.*}}func9(int)' uses aspect 'cpu' not listed in its 'sycl::device_has' attribute}}
[[sycl::device_has(sycl::aspect::fp64)]] int func9(int a) { return func8(a, 1); }

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() { int a = func1(1); });
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'sycl::device_has' attribute}}
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp64)]] {
      int a = func2(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp16)]] {
      int a = func2(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp16, sycl::aspect::fp64)]] {
      int a = func2(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp64, sycl::aspect::fp16)]] {
      int a = func2(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'sycl::device_has' attribute}}
    // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'sycl::device_has' attribute}}
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::gpu)]] {
      int a = func4(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'sycl::device_has' attribute}}
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp16)]] {
      int a = func4(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'sycl::device_has' attribute}}
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp64)]] {
      int a = func4(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp16, sycl::aspect::fp64)]] {
      int a = func4(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp64, sycl::aspect::fp16)]] {
      int a = func4(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::gpu, sycl::aspect::fp16, sycl::aspect::fp64)]] {
      int a = func4(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp16, sycl::aspect::fp64, sycl::aspect::gpu)]] {
      int a = func4(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() { int a = func9(1); });
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'sycl::device_has' attribute}}
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp64)]] {
      int a = func8(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::cpu)]] {
      int a = func8(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::fp64)]] {
      int a = func8(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp64, sycl::aspect::cpu)]] {
      int a = func8(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'sycl::device_has' attribute}}
    // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'sycl::device_has' attribute}}
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::gpu)]] {
      int a = func7(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'sycl::device_has' attribute}}
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::cpu)]] {
      int a = func7(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'sycl::device_has' attribute}}
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp64)]] {
      int a = func7(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::fp64)]] {
      int a = func7(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::fp64, sycl::aspect::cpu)]] {
      int a = func7(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::gpu, sycl::aspect::cpu, sycl::aspect::fp64)]] {
      int a = func7(1, 2);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::fp64, sycl::aspect::gpu)]] {
      int a = func7(1, 2);
    });
  });
}
